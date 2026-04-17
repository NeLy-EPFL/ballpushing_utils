#!/usr/bin/env python3
"""
Create heatmaps of fly positions aligned to template, grouped by condition.
Uses template matching to align each fly's video to the template.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
import argparse


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
    """Detect arena position and scale in a video using the last frame."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Add padding
    frame_padded, padding = add_padding(frame, padding_percent=padding_percent)

    # Binarize
    frame_binary = binarize_video_frame(frame_padded, method="otsu")

    # Match
    match_result = match_binary_multiscale(frame_binary, template_binary, scale_range=(0.3, 0.8), scale_steps=60)

    if match_result is None or match_result["score"] < 0.7:
        return None

    # Location in padded frame
    padded_loc = match_result["location"]

    # Location in original frame (accounting for padding)
    original_loc = (padded_loc[0] - padding[2], padded_loc[1] - padding[0])

    # Template size in video coordinates
    template_w, template_h = match_result["size"]

    return {
        "score": match_result["score"],
        "scale": match_result["scale"],
        "arena_x": original_loc[0],
        "arena_y": original_loc[1],
        "arena_width": template_w,
        "arena_height": template_h,
    }


def transform_fly_positions_to_template(
    fly_data, arena_params, template_shape, x_col="x_thorax_fly_0", y_col="y_thorax_fly_0"
):
    """Transform fly positions from video coordinates to template coordinates."""
    df = fly_data.copy()

    x_video = df[x_col].values
    y_video = df[y_col].values

    # Transform to arena-relative coordinates
    x_arena = x_video - arena_params["arena_x"]
    y_arena = y_video - arena_params["arena_y"]

    # Scale to template size
    scale = arena_params["scale"]
    x_template = x_arena / scale
    y_template = y_arena / scale

    df["x_template"] = x_template
    df["y_template"] = y_template

    # Filter out points outside template bounds
    template_h, template_w = template_shape[:2]
    mask = (x_template >= 0) & (x_template < template_w) & (y_template >= 0) & (y_template < template_h)
    df_filtered = df[mask].copy()

    return df_filtered


def filter_data_from_first_movement(df, x_col, fly_col="fly", threshold=100, n_avg=100):
    """
    Filter data to keep only points after fly first moves >threshold from starting position.

    Args:
        df: DataFrame with fly tracking data
        x_col: Column name for x coordinates
        fly_col: Column name for fly identifier
        threshold: Distance threshold in pixels (default: 100)
        n_avg: Number of initial points to average for starting position (default: 100)

    Returns:
        Filtered DataFrame
    """
    filtered_dfs = []

    for fly_id in df[fly_col].unique():
        fly_data = df[df[fly_col] == fly_id].copy()

        # Sort by frame or index to ensure temporal order
        if "frame" in fly_data.columns:
            fly_data = fly_data.sort_values("frame")
        else:
            fly_data = fly_data.sort_index()

        # Calculate starting position (average of first n_avg points)
        n_use = min(n_avg, len(fly_data))
        if n_use == 0:
            continue

        x_start = fly_data[x_col].iloc[:n_use].mean()

        # Calculate distance from start
        fly_data["distance_from_start"] = np.abs(fly_data[x_col] - x_start)

        # Find first time distance exceeds threshold
        threshold_mask = fly_data["distance_from_start"] > threshold

        if threshold_mask.any():
            first_threshold_idx = threshold_mask.idxmax()
            # Keep all data from this point onwards
            fly_data_filtered = fly_data.loc[first_threshold_idx:]
            filtered_dfs.append(fly_data_filtered)
        # If fly never exceeds threshold, don't include this fly

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def get_video_path_from_fly_id(fly_id, videos_base):
    """Construct video path from fly ID."""
    fly_id_str = str(fly_id)
    parts = fly_id_str.split("_")

    arena = None
    side = None
    for i, part in enumerate(parts):
        if "arena" in part.lower():
            arena = part
            if i + 1 < len(parts):
                side = parts[i + 1]
            break

    if arena is None or side is None:
        return None

    video_path = videos_base / arena / side / f"{side}.mp4"

    if not video_path.exists():
        return None

    return video_path


def get_ball_position_in_template(data, arena_params, template_shape, condition):
    """
    Get the median ball position in template coordinates.

    Args:
        data: DataFrame with ball position data
        arena_params: Arena transformation parameters
        template_shape: Template image shape
        condition: Condition name (to determine which ball to use)

    Returns:
        Tuple of (x_template, y_template) or None if ball not found
    """
    # Determine which ball to use based on condition
    # Control/no pretraining: ball_0, Experimental/pretrained: ball_1
    if condition in ["control", "n"]:
        ball_x_col = "x_centre_ball_0"
        ball_y_col = "y_centre_ball_0"
    else:
        ball_x_col = "x_centre_ball_1"
        ball_y_col = "y_centre_ball_1"

    # Check if ball columns exist
    if ball_x_col not in data.columns or ball_y_col not in data.columns:
        return None

    # Get ball positions (use median to get typical position)
    ball_x_video = data[ball_x_col].median()
    ball_y_video = data[ball_y_col].median()

    if pd.isna(ball_x_video) or pd.isna(ball_y_video):
        return None

    # Transform to template coordinates
    x_arena = ball_x_video - arena_params["arena_x"]
    y_arena = ball_y_video - arena_params["arena_y"]

    scale = arena_params["scale"]
    x_template = x_arena / scale
    y_template = y_arena / scale

    # Check if within template bounds
    template_h, template_w = template_shape[:2]
    if 0 <= x_template < template_w and 0 <= y_template < template_h:
        return (x_template, y_template)

    return None


def create_heatmap_on_template(data, template, template_binary, bins=100, blur_sigma=2.0, alpha=0.7):
    """Create a heatmap overlaid on template image, masked to white areas only.

    Heatmap is computed at 'bins' resolution then upscaled to template size to preserve mask details.
    """
    # Create per-fly heatmaps
    template_h, template_w = template.shape[:2]

    x_edges = np.linspace(0, template_w, bins + 1)
    y_edges = np.linspace(0, template_h, bins + 1)

    flies = data["fly"].unique()
    heatmaps = []

    for fly in flies:
        fly_data = data[data["fly"] == fly]
        x = fly_data["x_template"].values
        y = fly_data["y_template"].values

        if len(x) == 0:
            continue

        heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        # Normalize by number of points for this fly
        if len(x) > 0:
            heatmap = heatmap / len(x)

        heatmaps.append(heatmap)

    if not heatmaps:
        return None

    # Average across flies
    avg_heatmap = np.mean(heatmaps, axis=0)

    # Apply Gaussian blur
    if blur_sigma > 0:
        avg_heatmap = gaussian_filter(avg_heatmap, sigma=blur_sigma)

    # Upscale heatmap to match template size (width, height = template_w, template_h)
    # histogram2d returns (x_bins, y_bins), so we need to transpose before resize
    heatmap_upscaled = cv2.resize(avg_heatmap.T, (template_w, template_h), interpolation=cv2.INTER_LINEAR)

    # Use original template binary as mask (already at full resolution)
    template_mask = template_binary > 0  # shape (template_h, template_w)

    # Now heatmap_upscaled is (template_h, template_w) after transpose
    heatmap_display = heatmap_upscaled.copy()
    heatmap_display[~template_mask] = np.nan

    # Also set zeros to NaN for transparency
    heatmap_display[heatmap_display == 0] = np.nan

    return heatmap_display


def main():
    parser = argparse.ArgumentParser(description="Create aligned heatmaps grouped by condition")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode: only process 2 flies per condition and save a debug heatmap for orientation check",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["F1_condition", "Pretraining"],
        default="Pretraining",
        help="Column to group flies by (default: Pretraining)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for heatmap (default: 100)",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=2.0,
        help="Gaussian blur sigma (default: 2.0)",
    )
    parser.add_argument(
        "--max-flies",
        type=int,
        default=None,
        help="Maximum number of flies to process (for testing)",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=95,
        help="Percentile to clip colormap at (default: 95 to reduce starting position dominance)",
    )

    args = parser.parse_args()

    # Configure matplotlib for editable PDF text
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    matplotlib.rcParams["font.family"] = "Arial"
    # Note: text.usetex = True requires LaTeX installation, skip for now

    # Paths
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    dataset_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251121_17_summary_F1_New_Data/fly_positions/pooled_fly_positions.feather"
    )
    videos_base = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked")
    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New/PR/Heatmaps")

    # make dir if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ALIGNED HEATMAPS BY {args.group_by.upper()}")
    print("=" * 70)

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

    # Check if grouping column exists
    if args.group_by not in df.columns:
        print(f"   ERROR: Column '{args.group_by}' not found in dataset!")
        return

    # Find position columns first (needed for filtering)
    x_col = None
    y_col = None
    possible_x_cols = ["x_thorax_fly_0", "x_thorax", "x_thorax_skeleton"]
    possible_y_cols = ["y_thorax_fly_0", "y_thorax", "y_thorax_skeleton"]

    for col in possible_x_cols:
        if col in df.columns:
            x_col = col
            break

    for col in possible_y_cols:
        if col in df.columns:
            y_col = col
            break

    if x_col is None or y_col is None:
        print(f"   ERROR: Could not find position columns!")
        return

    print(f"\n3. Using position columns: x_col={x_col}, y_col={y_col}")

    # Filter data: keep only from first time fly moves >100px from start
    print("\n4. Filtering data from first movement >100px from start...")
    df = filter_data_from_first_movement(df, x_col, fly_col="fly", threshold=100, n_avg=100)
    print(f"   Dataset shape after movement filter: {df.shape}")
    print(f"   Number of unique flies after filter: {df['fly'].nunique()}")

    if len(df) == 0:
        print("   ERROR: No data remaining after filter!")
        return

    # Get unique conditions
    conditions = sorted(df[args.group_by].dropna().unique())
    print(f"\n5. Found {len(conditions)} conditions in '{args.group_by}':")
    for condition in conditions:
        n_flies = df[df[args.group_by] == condition]["fly"].nunique()
        print(f"   - {condition}: {n_flies} flies")

    # Process flies and align to template
    print("\n6. Processing flies and aligning to template...")

    aligned_data_by_condition = {condition: [] for condition in conditions}

    fly_ids = df["fly"].unique()
    if args.test_mode:
        # Only pick 2 flies per condition for quick test
        test_fly_ids = []
        for condition in conditions:
            cond_fly_ids = df[df[args.group_by] == condition]["fly"].unique()
            test_fly_ids.extend(cond_fly_ids[:2])
        fly_ids = np.array(test_fly_ids)
        print(f"\nTEST MODE: Only processing {len(fly_ids)} flies (2 per condition)")
    elif args.max_flies:
        fly_ids = fly_ids[: args.max_flies]

    processed_count = 0
    failed_count = 0

    for i, fly_id in enumerate(fly_ids, 1):
        if i % 10 == 0:
            print(f"   Processing fly {i}/{len(fly_ids)}...")

        fly_data = df[df["fly"] == fly_id].copy()

        # Get condition
        condition = fly_data[args.group_by].iloc[0]
        if pd.isna(condition):
            continue

        # Get video path
        video_path = get_video_path_from_fly_id(fly_id, videos_base)
        if video_path is None:
            failed_count += 1
            continue

        # Detect arena
        arena_params = detect_arena_in_video(video_path, template_binary)
        if arena_params is None:
            failed_count += 1
            continue

        # Transform positions
        fly_data_transformed = transform_fly_positions_to_template(
            fly_data, arena_params, template.shape, x_col=x_col, y_col=y_col
        )

        if len(fly_data_transformed) == 0:
            failed_count += 1
            continue

        aligned_data_by_condition[condition].append(fly_data_transformed)
        processed_count += 1

    print(f"\n   Successfully processed: {processed_count}/{len(fly_ids)} flies")
    print(f"   Failed: {failed_count}/{len(fly_ids)} flies")

    # Concatenate data for each condition
    print("\n7. Creating heatmaps...")
    condition_data = {}
    for condition in conditions:
        if aligned_data_by_condition[condition]:
            condition_df = pd.concat(aligned_data_by_condition[condition], ignore_index=True)
            condition_data[condition] = condition_df
            n_flies = condition_df["fly"].nunique()
            n_points = len(condition_df)
            print(f"   {condition}: {n_flies} flies, {n_points:,} points")

    # Compute common ball position across all conditions (using ball_1 only)
    print("\n   Computing common ball position across all data (ball_1)...")
    common_ball_position = None
    if condition_data:
        # Combine all condition data
        all_data = pd.concat(condition_data.values(), ignore_index=True)
        # Get a sample fly to obtain arena parameters
        sample_fly = all_data["fly"].iloc[0]
        sample_video = get_video_path_from_fly_id(sample_fly, videos_base)
        if sample_video:
            sample_arena_params = detect_arena_in_video(sample_video, template_binary)
            if sample_arena_params:
                # Use only ball_1
                ball_x_col = "x_centre_ball_1"
                ball_y_col = "y_centre_ball_1"
                if ball_x_col in all_data.columns and ball_y_col in all_data.columns:
                    ball_x_video = all_data[ball_x_col].median()
                    ball_y_video = all_data[ball_y_col].median()
                    if not pd.isna(ball_x_video) and not pd.isna(ball_y_video):
                        # Transform to template coordinates
                        x_arena = ball_x_video - sample_arena_params["arena_x"]
                        y_arena = ball_y_video - sample_arena_params["arena_y"]
                        scale = sample_arena_params["scale"]
                        x_template = x_arena / scale
                        y_template = y_arena / scale
                        # Check if within template bounds
                        template_h, template_w = template.shape[:2]
                        if 0 <= x_template < template_w and 0 <= y_template < template_h:
                            common_ball_position = (x_template, y_template)
                            print(f"   Common ball position: ({x_template:.1f}, {y_template:.1f})")

    # Create figure with subplots
    n_conditions = len(condition_data)
    if n_conditions == 0:
        print("\n   ERROR: No data to plot!")
        return

    # Determine subplot layout
    if n_conditions <= 3:
        n_cols = n_conditions
        n_rows = 1
    else:
        n_cols = 3
        n_rows = (n_conditions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 10 * n_rows))

    # Handle single subplot case
    if n_conditions == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # Pre-compute all heatmaps to find global vmax
    print("\n   Computing heatmaps...")
    heatmaps = {}
    global_vmax = 0

    for condition, data in condition_data.items():
        heatmap = create_heatmap_on_template(
            data, template, template_binary, bins=args.bins, blur_sigma=args.blur_sigma
        )
        heatmaps[condition] = heatmap
        if heatmap is not None:
            condition_max = np.nanmax(heatmap)
            if condition_max > global_vmax:
                global_vmax = condition_max

    print(f"   Global vmax for consistent scaling: {global_vmax:.6f}")

    # Create multiple scale versions to handle starting position hotspot
    scale_configs = [
        {
            "name": "linear_clipped",
            "vmax_percentile": args.clip_percentile,
            "transform": None,
            "label": f"Avg. Normalized Density (fraction/bin)\n(clipped at {args.clip_percentile}th percentile)",
        },
        {
            "name": "log",
            "vmax_percentile": None,
            "transform": "log",
            "label": "Log₁₀ Avg. Normalized Density\n(fraction/bin)",
        },
        {
            "name": "sqrt",
            "vmax_percentile": None,
            "transform": "sqrt",
            "label": "√(Avg. Normalized Density)\n(√[fraction/bin])",
        },
    ]

    # In test mode, only plot the first (linear) scale and save a debug image for orientation
    if args.test_mode:
        print("\nTEST MODE: Plotting debug heatmap for orientation check...")
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(template_rgb, aspect="auto", alpha=1.0)
        # Just use the first condition's heatmap
        for condition, heatmap in heatmaps.items():
            if heatmap is not None:
                cmap = plt.get_cmap("viridis")
                cmap.set_bad(alpha=0)
                im = ax.imshow(
                    heatmap,
                    extent=(0, template.shape[1], template.shape[0], 0),
                    origin="upper",
                    cmap=cmap,
                    alpha=0.7,
                    interpolation="bilinear",
                )
                ax.set_title(f"DEBUG: {condition}", fontsize=16, fontweight="bold")
                plt.colorbar(im, ax=ax, label="Debug Density", fraction=0.046, pad=0.04)
                break  # Only plot one for speed
        plt.tight_layout()
        debug_path = output_dir / "debug_heatmap_orientation.png"
        fig.savefig(debug_path, dpi=200, bbox_inches="tight")
        print(f"   Saved debug orientation heatmap to: {debug_path}")
        plt.close()
        print("\nTest mode complete. Exiting early.")
        return

    # Normal plotting for all scale configs
    for scale_config in scale_configs:
        scale_type = scale_config["name"]
        print(f"\n   Creating {scale_type} scale version...")

        # Calculate vmax for this scale type
        if scale_config["vmax_percentile"] is not None:
            # Use percentile clipping
            all_values = []
            for heatmap in heatmaps.values():
                if heatmap is not None:
                    all_values.extend(heatmap[~np.isnan(heatmap)].flatten())
            if all_values:
                scale_vmax = np.percentile(all_values, scale_config["vmax_percentile"])
                print(f"     Using {scale_config['vmax_percentile']}th percentile: {scale_vmax:.6f}")
            else:
                scale_vmax = global_vmax
        else:
            scale_vmax = None  # Auto-scale

        # Create fresh figure for each scale type
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 10 * n_rows))

        # Handle single subplot case
        if n_conditions == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if n_conditions > 1 else axes

        for idx, (condition, data) in enumerate(condition_data.items()):
            ax = axes_flat[idx]

            # Show template as background (fully opaque to keep it intact)
            ax.imshow(template_rgb, aspect="auto", alpha=1.0)

            # Get pre-computed heatmap
            heatmap = heatmaps[condition]

            if heatmap is not None:
                cmap = plt.get_cmap("viridis")
                cmap.set_bad(alpha=0)

                # Apply transformation if requested
                heatmap_display = heatmap.copy()
                mask = np.isnan(heatmap_display) | (heatmap_display == 0)

                if scale_config["transform"] == "log":
                    # Log scale with small epsilon
                    heatmap_display[~mask] = np.log10(heatmap_display[~mask] + 1e-10)
                    vmin = None
                    vmax = None
                elif scale_config["transform"] == "sqrt":
                    # Square root scale
                    heatmap_display[~mask] = np.sqrt(heatmap_display[~mask])
                    vmin = 0
                    vmax = np.sqrt(scale_vmax) if scale_vmax is not None else None
                else:
                    # Linear (possibly clipped)
                    vmin = 0
                    vmax = scale_vmax if scale_vmax is not None else global_vmax

                # Restore NaNs for transparency
                heatmap_display[mask] = np.nan

                im = ax.imshow(
                    heatmap_display,
                    extent=(0, template.shape[1], template.shape[0], 0),
                    origin="upper",
                    cmap=cmap,
                    alpha=0.7,
                    interpolation="bilinear",
                    vmin=vmin,
                    vmax=vmax,
                )

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, label=scale_config["label"], fraction=0.046, pad=0.04)

            # Add title and stats
            n_flies = data["fly"].nunique()
            n_points = len(data)
            ax.set_title(f"{condition}\n{n_flies} flies, {n_points:,} points", fontsize=14, fontweight="bold")

            ax.set_xlim(0, template.shape[1])
            ax.set_ylim(template.shape[0], 0)
            ax.set_xlabel("X (template pixels)", fontsize=12)
            ax.set_ylabel("Y (template pixels)", fontsize=12)
            ax.set_aspect("equal")

        # Hide empty subplots
        for idx in range(n_conditions, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()

        # Save figure (standard version without proximity circle)
        output_filename = f"aligned_heatmaps_by_{args.group_by}_{scale_type}.png"
        output_path = output_dir / output_filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"   Saved {scale_type} scale heatmap to: {output_path}")

        plt.close()

        # Create version with proximity circles if common ball position is available
        if common_ball_position:
            # Create versions with different proximity radii
            proximity_radii = [70, 140, 200]

            for proximity_radius_px in proximity_radii:
                print(f"   Creating {scale_type} scale version with {proximity_radius_px}px proximity circle...")
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 10 * n_rows))

                # Handle single subplot case
                if n_conditions == 1:
                    axes = np.array([axes])
                axes_flat = axes.flatten() if n_conditions > 1 else axes

                for idx, (condition, data) in enumerate(condition_data.items()):
                    ax = axes_flat[idx]

                    # Show template as background
                    ax.imshow(template_rgb, aspect="auto", alpha=1.0, rasterized=True)

                    # Get pre-computed heatmap
                    heatmap = heatmaps[condition]

                    if heatmap is not None:
                        cmap = plt.get_cmap("viridis")
                        cmap.set_bad(alpha=0)

                        # Apply transformation if requested
                        heatmap_display = heatmap.copy()
                        mask = np.isnan(heatmap_display) | (heatmap_display == 0)

                        if scale_config["transform"] == "log":
                            heatmap_display[~mask] = np.log10(heatmap_display[~mask] + 1e-10)
                            vmin = None
                            vmax = None
                        elif scale_config["transform"] == "sqrt":
                            heatmap_display[~mask] = np.sqrt(heatmap_display[~mask])
                            vmin = 0
                            vmax = np.sqrt(scale_vmax) if scale_vmax is not None else None
                        else:
                            vmin = 0
                            vmax = scale_vmax if scale_vmax is not None else global_vmax

                        heatmap_display[mask] = np.nan

                        im = ax.imshow(
                            heatmap_display,
                            extent=(0, template.shape[1], template.shape[0], 0),
                            origin="upper",
                            cmap=cmap,
                            alpha=0.7,
                            interpolation="bilinear",
                            vmin=vmin,
                            vmax=vmax,
                            rasterized=True,
                        )

                        cbar = plt.colorbar(im, ax=ax, label=scale_config["label"], fraction=0.046, pad=0.04)

                    # Draw proximity circle using common ball position
                    ball_x, ball_y = common_ball_position

                    circle = Circle(
                        (ball_x, ball_y),
                        proximity_radius_px,
                        color="red",
                        fill=False,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        label=f"{proximity_radius_px}px proximity",
                    )
                    ax.add_patch(circle)

                    # Add small dot at ball center
                    ax.plot(ball_x, ball_y, "r+", markersize=10, markeredgewidth=2, alpha=0.8)

                    # Add title and stats
                    n_flies = data["fly"].nunique()

                    # Format condition name
                    condition_display = condition
                    if args.group_by == "Pretraining":
                        condition_display = {"n": "Naive", "y": "Pretrained"}.get(condition, condition)

                    title = f"{condition_display}\n{n_flies} flies"
                    title += f"\n(red circle: ball proximity region, threshold: {proximity_radius_px}px)"
                    ax.set_title(title, fontsize=14, fontweight="bold")

                    ax.set_xlim(0, template.shape[1])
                    ax.set_ylim(template.shape[0], 0)
                    ax.set_xlabel("X (template pixels)", fontsize=12)
                    ax.set_ylabel("Y (template pixels)", fontsize=12)
                    ax.set_aspect("equal")

                # Hide empty subplots
                for idx in range(n_conditions, len(axes_flat)):
                    axes_flat[idx].axis("off")

                plt.tight_layout()

                # Save figure with proximity circles as PNG
                output_filename_prox = (
                    f"aligned_heatmaps_by_{args.group_by}_{scale_type}_with_proximity_{proximity_radius_px}px.png"
                )
                output_path_prox = output_dir / output_filename_prox
                fig.savefig(output_path_prox, dpi=300, bbox_inches="tight")
                print(
                    f"   Saved {scale_type} scale heatmap with {proximity_radius_px}px proximity to: {output_path_prox}"
                )

                # Save as PDF (vector text/axes, rasterized images) - only for linear_clipped
                if scale_type == "linear_clipped":
                    output_filename_pdf = (
                        f"aligned_heatmaps_by_{args.group_by}_{scale_type}_with_proximity_{proximity_radius_px}px.pdf"
                    )
                    output_path_pdf = output_dir / output_filename_pdf
                    fig.savefig(output_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
                    print(
                        f"   Saved {scale_type} scale heatmap with {proximity_radius_px}px proximity (PDF) to: {output_path_pdf}"
                    )

                plt.close()

    print("\n8. COMPLETE!")
    print("=" * 70)
    print(f"Generated linear_clipped, log, and sqrt scale heatmaps for {args.group_by}")
    print(f"Clipping percentile: {args.clip_percentile}th")
    print("=" * 70)


if __name__ == "__main__":
    main()
