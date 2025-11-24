#!/usr/bin/env python3
"""
Create heatmaps with template background, masking non-arena areas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib import colors
import matplotlib.patches as mpatches


def find_white_area_bounds(template_binary):
    """
    Find the bounding box of the white (arena) area in the template.

    Args:
        template_binary: Binary template (255=arena, 0=background)

    Returns:
        Dictionary with x_min, x_max, y_min, y_max of white area
    """
    # Find all white pixels
    white_pixels = np.where(template_binary > 0)

    if len(white_pixels[0]) == 0:
        return None

    y_coords = white_pixels[0]
    x_coords = white_pixels[1]

    return {
        "x_min": int(np.min(x_coords)),
        "x_max": int(np.max(x_coords)),
        "y_min": int(np.min(y_coords)),
        "y_max": int(np.max(y_coords)),
        "width": int(np.max(x_coords) - np.min(x_coords)),
        "height": int(np.max(y_coords) - np.min(y_coords)),
    }


def create_masked_heatmap(
    data, x_col, y_col, template, arena_mask, bins=100, blur_sigma=2.0, clip_percent=0.05, colormap="hot", alpha=0.7
):
    """
    Create a heatmap overlaid on the template, with data only in arena regions.

    Args:
        data: DataFrame with position data in template coordinates
        x_col: Column name for x coordinates (in template space)
        y_col: Column name for y coordinates (in template space)
        template: Template image (BGR)
        arena_mask: Binary mask (True=arena, False=background)
        bins: Number of bins for histogram
        blur_sigma: Gaussian blur sigma
        clip_percent: Percentage of max value to clip for display
        colormap: Matplotlib colormap name
        alpha: Alpha value for heatmap overlay

    Returns:
        Figure with heatmap
    """
    # Remove NaN values
    clean_data = data[[x_col, y_col]].dropna()

    if clean_data.empty:
        print("Warning: No valid data for heatmap")
        return None

    # Get template dimensions
    template_h, template_w = template.shape[:2]

    # Create 2D histogram
    x = clean_data[x_col].values
    y = clean_data[y_col].values

    # Define bins based on template size
    x_edges = np.linspace(0, template_w, bins + 1)
    y_edges = np.linspace(0, template_h, bins + 1)

    heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Normalize by number of points
    if len(x) > 0:
        heatmap = heatmap / len(x)

    # Apply Gaussian blur
    if blur_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

    # Mask areas outside the arena
    # Resize arena mask to match heatmap bins
    arena_mask_resized = cv2.resize(arena_mask.astype(np.uint8), (bins, bins), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )

    # Apply mask: set non-arena areas to NaN (will be transparent)
    heatmap_masked = heatmap.copy()
    heatmap_masked[~arena_mask_resized.T] = np.nan

    # Clip low values for better visualization
    vmax = np.nanmax(heatmap_masked)
    threshold = vmax * clip_percent
    heatmap_masked[heatmap_masked < threshold] = np.nan

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    # Show template as background
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    ax.imshow(template_rgb, extent=[0, template_w, template_h, 0], aspect="auto")

    # Overlay heatmap
    cmap = plt.get_cmap(colormap)
    cmap.set_bad(alpha=0)  # Make NaN values transparent

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
    cbar = plt.colorbar(im, ax=ax, label="Normalized Density", fraction=0.046, pad=0.04)

    # Add sample size annotation
    n_flies = len(data["fly"].unique()) if "fly" in data.columns else "unknown"
    n_points = len(clean_data)
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0, template_w)
    ax.set_ylim(template_h, 0)
    ax.set_xlabel("X (template pixels)", fontsize=12)
    ax.set_ylabel("Y (template pixels)", fontsize=12)
    ax.set_aspect("equal")

    return fig, ax


def create_average_fly_masked_heatmap(
    data, x_col, y_col, template, arena_mask, bins=100, blur_sigma=2.0, clip_percent=0.05, colormap="hot", alpha=0.7
):
    """
    Create a heatmap with per-fly averaging, overlaid on template with masking.

    Args:
        data: DataFrame with position data in template coordinates
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        template: Template image (BGR)
        arena_mask: Binary mask (True=arena, False=background)
        bins: Number of bins for histogram
        blur_sigma: Gaussian blur sigma
        clip_percent: Percentage of max value to clip
        colormap: Matplotlib colormap name
        alpha: Alpha value for heatmap overlay

    Returns:
        Figure with heatmap
    """
    if "fly" not in data.columns:
        print("Warning: 'fly' column not found, using simple heatmap")
        return create_masked_heatmap(
            data, x_col, y_col, template, arena_mask, bins, blur_sigma, clip_percent, colormap, alpha
        )

    # Get template dimensions
    template_h, template_w = template.shape[:2]

    # Define bins
    x_edges = np.linspace(0, template_w, bins + 1)
    y_edges = np.linspace(0, template_h, bins + 1)

    # Calculate per-fly heatmaps
    flies = data["fly"].unique()
    heatmaps = []

    for fly in flies:
        fly_data = data[data["fly"] == fly]
        x = fly_data[x_col].dropna().values
        y = fly_data[y_col].dropna().values

        if len(x) == 0:
            continue

        heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        # Normalize by number of points for this fly
        if len(x) > 0:
            heatmap = heatmap / len(x)

        heatmaps.append(heatmap)

    if not heatmaps:
        print("Warning: No valid fly data for heatmap")
        return None

    # Average across flies
    avg_heatmap = np.mean(heatmaps, axis=0)

    # Apply Gaussian blur
    if blur_sigma > 0:
        avg_heatmap = gaussian_filter(avg_heatmap, sigma=blur_sigma)

    # Mask areas outside the arena
    arena_mask_resized = cv2.resize(arena_mask.astype(np.uint8), (bins, bins), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )

    # Apply mask
    heatmap_masked = avg_heatmap.copy()
    heatmap_masked[~arena_mask_resized.T] = np.nan

    # Clip low values
    vmax = np.nanmax(heatmap_masked)
    threshold = vmax * clip_percent
    heatmap_masked[heatmap_masked < threshold] = np.nan

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    # Show template as background
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    ax.imshow(template_rgb, extent=[0, template_w, template_h, 0], aspect="auto")

    # Overlay heatmap
    cmap = plt.get_cmap(colormap)
    cmap.set_bad(alpha=0)  # Make NaN values transparent

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
    n_flies = len(flies)
    n_points = len(data[[x_col, y_col]].dropna())
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0, template_w)
    ax.set_ylim(template_h, 0)
    ax.set_xlabel("X (template pixels)", fontsize=12)
    ax.set_ylabel("Y (template pixels)", fontsize=12)
    ax.set_aspect("equal")

    return fig, ax


def create_heatmap_with_template_overlay(
    data,
    x_col,
    y_col,
    template,
    arena_mask,
    template_binary,
    scale_factor=None,
    video_shape=None,
    bins=100,
    blur_sigma=2.0,
    clip_percent=0.0,
    colormap="hot",
    alpha=0.7,
    chamber_offset_x=20,
    chamber_offset_y=20,
):
    """
    Create a heatmap overlaid on template, properly scaling and aligning template to data coordinates.
    The chamber origin (white area bottom-left + offsets) is set to (0,0) in data coordinates.

    Args:
        data: DataFrame with position data in video/centered coordinates
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        template: Template image (BGR)
        arena_mask: Binary mask (True=arena, False=background)
        template_binary: Binary template (255=arena, 0=background)
        scale_factor: Scale to apply to template (if None, auto-detect from video_shape)
        video_shape: Tuple of (height, width) of video for auto-scaling
        bins: Number of bins for histogram
        blur_sigma: Gaussian blur sigma
        clip_percent: Percentage of max value to clip (0 = no clipping)
        colormap: Matplotlib colormap name
        alpha: Alpha value for heatmap overlay
        chamber_offset_x: X offset from white area bottom-left to chamber center (default: 20px)
        chamber_offset_y: Y offset from white area bottom-left to chamber center (default: -20px down)

    Returns:
        Tuple of (fig, ax) or None
    """
    if "fly" not in data.columns:
        print("Warning: 'fly' column not found")
        return None

    # Remove NaN values
    clean_data = data[[x_col, y_col, "fly"]].dropna()
    if clean_data.empty:
        print("Warning: No valid data for heatmap")
        return None

    # Get template dimensions
    template_h, template_w = template.shape[:2]

    # Find white area bounds in original template
    white_bounds = find_white_area_bounds(template_binary)
    if white_bounds is None:
        print("Warning: No white area found in template")
        return None

    print(f"Template white area bounds: {white_bounds}")

    # Determine scale factor if not provided
    if scale_factor is None:
        # Use a typical scale based on video dimensions (if provided)
        # or estimate from data range
        if video_shape is not None:
            # Estimate scale: assume template should roughly match video size
            scale_factor = min(video_shape[1] / template_w, video_shape[0] / template_h) * 0.6
        else:
            # Estimate from data range
            data_x_range = clean_data[x_col].max() - clean_data[x_col].min()
            data_y_range = clean_data[y_col].max() - clean_data[y_col].min()
            scale_factor = min(data_x_range / template_w, data_y_range / template_h) * 1.5

    print(f"Using template scale factor: {scale_factor:.3f}")

    # Scale template and masks
    scaled_w = int(template_w * scale_factor)
    scaled_h = int(template_h * scale_factor)
    template_scaled = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    arena_mask_scaled = cv2.resize(
        arena_mask.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Scale white bounds
    white_bounds_scaled = {
        "x_min": int(white_bounds["x_min"] * scale_factor),
        "x_max": int(white_bounds["x_max"] * scale_factor),
        "y_min": int(white_bounds["y_min"] * scale_factor),
        "y_max": int(white_bounds["y_max"] * scale_factor),
    }

    # Calculate chamber origin in scaled template coordinates
    # Chamber is at bottom-left of white area + offsets
    chamber_x_in_template = white_bounds_scaled["x_min"] + chamber_offset_x
    chamber_y_in_template = white_bounds_scaled["y_max"] - chamber_offset_y  # y_max is bottom (image coords)

    print(f"Chamber origin in scaled template: ({chamber_x_in_template}, {chamber_y_in_template})")

    # Calculate template position in data coordinates
    # Chamber origin should be at (0, 0) in data coordinates
    template_x0 = -chamber_x_in_template
    template_y0 = -chamber_y_in_template
    template_x1 = template_x0 + scaled_w
    template_y1 = template_y0 + scaled_h

    print(f"Template extent in data coords: x=[{template_x0}, {template_x1}], y=[{template_y0}, {template_y1}]")

    # Calculate per-fly heatmaps in data coordinates
    x_min = clean_data[x_col].min()
    x_max = clean_data[x_col].max()
    y_min = clean_data[y_col].min()
    y_max = clean_data[y_col].max()

    print(f"Data extent: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

    # Define bins in data space
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    # Calculate per-fly heatmaps
    flies = clean_data["fly"].unique()
    heatmaps = []

    for fly in flies:
        fly_data = clean_data[clean_data["fly"] == fly]
        x = fly_data[x_col].values
        y = fly_data[y_col].values

        if len(x) == 0:
            continue

        heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        # Normalize by number of points for this fly
        if len(x) > 0:
            heatmap = heatmap / len(x)

        heatmaps.append(heatmap)

    if not heatmaps:
        print("Warning: No valid fly data for heatmap")
        return None

    # Average across flies
    avg_heatmap = np.mean(heatmaps, axis=0)

    # Apply Gaussian blur
    if blur_sigma > 0:
        avg_heatmap = gaussian_filter(avg_heatmap, sigma=blur_sigma)

    # Apply clipping if requested
    if clip_percent > 0:
        vmax = np.nanmax(avg_heatmap)
        threshold = vmax * clip_percent
        avg_heatmap[avg_heatmap < threshold] = 0

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Show template as background (flip Y to match image coordinates)
    template_rgb = cv2.cvtColor(template_scaled, cv2.COLOR_BGR2RGB)
    ax.imshow(
        template_rgb, extent=[template_x0, template_x1, template_y1, template_y0], aspect="auto", zorder=0, alpha=0.6
    )

    # Overlay heatmap
    cmap = plt.get_cmap(colormap)
    cmap.set_bad(alpha=0)  # Make zero/NaN values transparent

    # Set zeros to NaN for transparency
    heatmap_display = avg_heatmap.copy()
    heatmap_display[heatmap_display == 0] = np.nan

    vmax = np.nanmax(heatmap_display)
    im = ax.imshow(
        heatmap_display.T,
        extent=[x_min, x_max, y_max, y_min],
        origin="upper",
        cmap=cmap,
        alpha=alpha,
        interpolation="bilinear",
        vmin=0,
        vmax=vmax,
        zorder=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Avg. Normalized Density (per fly)", fraction=0.046, pad=0.04)

    # Add sample size annotation
    n_flies = len(flies)
    n_points = len(clean_data)
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add chamber origin marker
    ax.plot(0, 0, "r+", markersize=15, markeredgewidth=2, label="Chamber Origin", zorder=3)
    ax.legend(loc="lower right")

    # Set axis limits to show both template and data
    ax.set_xlim(min(template_x0, x_min), max(template_x1, x_max))
    ax.set_ylim(max(template_y1, y_max), min(template_y0, y_min))
    ax.set_xlabel("X Position (pixels)", fontsize=12)
    ax.set_ylabel("Y Position (pixels)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=2)

    return fig, ax


def test_masked_heatmap():
    """Test the masked heatmap function with sample data."""
    from arena_alignment import load_template, detect_arena_in_video, transform_coordinates_to_template

    # Load template
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    template, template_binary, arena_mask = load_template(template_path)

    print(f"Template shape: {template.shape}")
    print(f"Arena mask coverage: {arena_mask.sum() / arena_mask.size * 100:.1f}%")

    # Generate some random test data in template space
    np.random.seed(42)
    n_flies = 5
    n_points_per_fly = 1000

    data_list = []
    for fly_id in range(n_flies):
        # Random positions within template bounds (biased towards center)
        x = np.random.normal(template.shape[1] / 2, template.shape[1] / 6, n_points_per_fly)
        y = np.random.normal(template.shape[0] / 2, template.shape[0] / 6, n_points_per_fly)

        # Clip to bounds
        x = np.clip(x, 0, template.shape[1] - 1)
        y = np.clip(y, 0, template.shape[0] - 1)

        data_list.append(pd.DataFrame({"fly": fly_id, "x_template": x, "y_template": y}))

    data = pd.concat(data_list, ignore_index=True)

    print(f"\nGenerated {len(data)} data points for {n_flies} flies")

    # Create masked heatmap
    print("\nCreating masked heatmap...")
    fig, ax = create_average_fly_masked_heatmap(
        data,
        "x_template",
        "y_template",
        template,
        arena_mask,
        bins=100,
        blur_sigma=2.0,
        clip_percent=0.05,
        colormap="hot",
        alpha=0.7,
    )

    if fig:
        output_dir = Path("/home/durrieu/ballpushing_utils/outputs/template_matching_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_masked_heatmap.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved test heatmap to: {output_path}")
        plt.close(fig)

    print("\nTest complete!")


if __name__ == "__main__":
    test_masked_heatmap()
