#!/usr/bin/env python3
"""
Create heatmaps of fly positions aligned to template for TNT experiments.
Uses template matching to align each fly's video to the template.
Supports multiple TNT modes: MB247, LC10-2, DDC (with optional pooled variants).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
import argparse
import yaml


# Import the helper functions from the control script
import sys

sys.path.insert(0, str(Path(__file__).parent))


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
    """Filter data to keep only points after fly first moves >threshold from starting position."""
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

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def get_video_path_from_fly_id(fly_id, videos_base):
    """Construct video path from fly ID.

    For TNT experiments, fly IDs have format: YYMMDD_F1_TNT_*_Videos_Checked_arena#_Side
    We need to extract the date prefix, arena, and side to construct the path.
    """
    fly_id_str = str(fly_id)
    parts = fly_id_str.split("_")

    arena = None
    side = None
    date_prefix = None

    # Find arena and side
    for i, part in enumerate(parts):
        if "arena" in part.lower():
            arena = part
            if i + 1 < len(parts):
                side = parts[i + 1]
            break

    if arena is None or side is None:
        return None

    # Extract date prefix (first part before _F1_)
    # Fly ID format: YYMMDD_F1_TNT_*_Videos_Checked_arena#_Side
    # We need to reconstruct: YYMMDD_F1_TNT_*_Videos_Checked
    try:
        # Find where "_Videos_Checked" appears
        videos_checked_idx = fly_id_str.find("_Videos_Checked")
        if videos_checked_idx != -1:
            date_prefix = fly_id_str[: videos_checked_idx + len("_Videos_Checked")]
        else:
            return None
    except:
        return None

    # Construct path: videos_base / date_prefix / arena / side / side.mp4
    video_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos") / date_prefix / arena / side / f"{side}.mp4"

    if not video_path.exists():
        return None

    return video_path


def get_ball_position_in_template(data, arena_params, template_shape, pretraining):
    """
    Get the median ball position in template coordinates for TNT experiments.

    Args:
        data: DataFrame with ball position data
        arena_params: Arena transformation parameters
        template_shape: Template image shape
        pretraining: Pretraining value ('n' or 'y')

    Returns:
        Tuple of (x_template, y_template) or None if ball not found
    """
    # For TNT experiments: n (no pretraining) uses ball_0, y (pretrained) uses ball_1
    if pretraining == "n":
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
    """Create a heatmap overlaid on template image, masked to white areas only."""
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

    # Upscale heatmap to match template size
    heatmap_upscaled = cv2.resize(avg_heatmap.T, (template_w, template_h), interpolation=cv2.INTER_LINEAR)

    # Use original template binary as mask
    template_mask = template_binary > 0

    # Apply mask
    heatmap_display = heatmap_upscaled.copy()
    heatmap_display[~template_mask] = np.nan
    heatmap_display[heatmap_display == 0] = np.nan

    return heatmap_display


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "analysis_config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Create aligned heatmaps for TNT experiments")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tnt_mb247", "tnt_lc10_2", "tnt_ddc", "tnt_mb247_pooled", "tnt_lc10_2_pooled", "tnt_ddc_pooled"],
        help="TNT experiment mode",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode: only process 2 flies per condition",
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
        help="Percentile to clip colormap at (default: 95)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    mode_config = config["analysis_modes"][args.mode]

    # Get paths from config - handle pooled modes
    is_pooled_mode = "primary_dataset" in mode_config

    # Determine output directory based on mode
    base_mode = args.mode.replace("_pooled", "")

    if base_mode == "tnt_mb247":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/MB247")
    elif base_mode == "tnt_lc10_2":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/LC10-2")
    elif base_mode == "tnt_ddc":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/DDC")
    else:
        print(f"ERROR: Unknown mode {args.mode}")
        return

    # Use the same template as control experiments
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")

    print("=" * 70)
    print(f"ALIGNED HEATMAPS FOR {args.mode.upper()}")
    print("=" * 70)

    # Load template
    print("\n1. Loading template...")
    template = cv2.imread(str(template_path))
    if template is None:
        print(f"   ERROR: Could not load template from {template_path}")
        return
    template_binary = binarize_template(template, invert_background=True)
    print(f"   Template shape: {template.shape}")

    # Load dataset - with pooling support
    print("\n2. Loading fly positions dataset...")

    if is_pooled_mode:
        # Pooled mode - load and combine multiple datasets
        primary_mode = mode_config["primary_dataset"]
        primary_config = config["analysis_modes"][primary_mode]

        # Load primary dataset
        print(f"   Loading primary dataset from: {primary_mode}")
        df_primary = pd.read_feather(Path(primary_config["fly_positions_path"]))
        print(f"   Primary dataset shape: {df_primary.shape}")

        # Get genotypes to pool and modes to pool from
        shared_genotypes = mode_config.get("shared_genotypes", [])
        pool_from_modes = mode_config.get("pool_from_modes", [])

        if shared_genotypes and pool_from_modes:
            # Find genotype column
            genotype_col_temp = None
            for col in df_primary.columns:
                if "genotype" in col.lower():
                    genotype_col_temp = col
                    break

            if genotype_col_temp:
                pooled_dfs = [df_primary]

                for pool_mode in pool_from_modes:
                    print(f"   Pooling {shared_genotypes} from: {pool_mode}")
                    pool_config = config["analysis_modes"][pool_mode]
                    pool_positions_path = pool_config.get("fly_positions_path")

                    if pool_positions_path:
                        try:
                            df_pool = pd.read_feather(Path(pool_positions_path))

                            # Find genotype column in pooled dataset
                            pool_genotype_col = None
                            for col in df_pool.columns:
                                if "genotype" in col.lower():
                                    pool_genotype_col = col
                                    break

                            if pool_genotype_col:
                                # Filter for shared genotypes
                                df_pool_filtered = df_pool[df_pool[pool_genotype_col].isin(shared_genotypes)].copy()

                                if not df_pool_filtered.empty:
                                    print(f"     Added {df_pool_filtered['fly'].nunique()} flies from {pool_mode}")
                                    pooled_dfs.append(df_pool_filtered)
                        except Exception as e:
                            print(f"     Warning: Could not load data from {pool_mode}: {e}")

                # Combine all datasets
                if len(pooled_dfs) > 1:
                    df = pd.concat(pooled_dfs, ignore_index=True)
                    print(f"   Combined dataset shape: {df.shape}")
                    print(f"   Total unique flies after pooling: {df['fly'].nunique()}")
                else:
                    df = df_primary
            else:
                print("   Warning: Could not find genotype column, using primary dataset only")
                df = df_primary
        else:
            df = df_primary
    else:
        # Regular mode - load single dataset
        fly_positions_path = Path(mode_config["fly_positions_path"])
        df = pd.read_feather(fly_positions_path)

    print(f"   Dataset shape: {df.shape}")
    print(f"   Number of unique flies: {df['fly'].nunique()}")

    # Auto-detect grouping columns
    genotype_col = None
    pretraining_col = None

    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
        if "pretrain" in col.lower():
            pretraining_col = col

    if genotype_col is None or pretraining_col is None:
        print(f"   ERROR: Could not find genotype or pretraining columns!")
        return

    print(f"   Using genotype column: {genotype_col}")
    print(f"   Using pretraining column: {pretraining_col}")

    # Find position columns
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

    # Filter data
    print("\n4. Filtering data from first movement >100px from start...")
    df = filter_data_from_first_movement(df, x_col, fly_col="fly", threshold=100, n_avg=100)
    print(f"   Dataset shape after movement filter: {df.shape}")
    print(f"   Number of unique flies after filter: {df['fly'].nunique()}")

    if len(df) == 0:
        print("   ERROR: No data remaining after filter!")
        return

    # Get unique genotypes and pretraining values
    genotypes = sorted(df[genotype_col].dropna().unique())
    pretrainings = sorted(df[pretraining_col].dropna().unique())

    print(f"\n5. Found {len(genotypes)} genotypes and {len(pretrainings)} pretraining conditions:")
    for genotype in genotypes:
        for pretraining in pretrainings:
            n_flies = df[(df[genotype_col] == genotype) & (df[pretraining_col] == pretraining)]["fly"].nunique()
            print(f"   - {genotype} + {pretraining}: {n_flies} flies")

    # Process flies and align to template
    print("\n6. Processing flies and aligning to template...")

    # Create dictionary to store data by (genotype, pretraining) combinations
    aligned_data_by_combo = {}
    for genotype in genotypes:
        for pretraining in pretrainings:
            aligned_data_by_combo[(genotype, pretraining)] = []

    fly_ids = df["fly"].unique()
    if args.test_mode:
        test_fly_ids = []
        for genotype in genotypes:
            for pretraining in pretrainings:
                combo_fly_ids = df[(df[genotype_col] == genotype) & (df[pretraining_col] == pretraining)][
                    "fly"
                ].unique()
                test_fly_ids.extend(combo_fly_ids[:2])
        fly_ids = np.array(test_fly_ids)
        print(f"\nTEST MODE: Only processing {len(fly_ids)} flies (2 per combination)")
    elif args.max_flies:
        fly_ids = fly_ids[: args.max_flies]

    processed_count = 0
    failed_count = 0

    for i, fly_id in enumerate(fly_ids, 1):
        if i % 10 == 0:
            print(f"   Processing fly {i}/{len(fly_ids)}...")

        fly_data = df[df["fly"] == fly_id].copy()

        # Get genotype and pretraining for this fly
        genotype = fly_data[genotype_col].iloc[0]
        pretraining = fly_data[pretraining_col].iloc[0]

        if pd.isna(genotype) or pd.isna(pretraining):
            continue

        # Get video path (videos_base not needed since path is extracted from fly ID)
        video_path = get_video_path_from_fly_id(fly_id, None)
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

        aligned_data_by_combo[(genotype, pretraining)].append(fly_data_transformed)
        processed_count += 1

    print(f"\n   Successfully processed: {processed_count}/{len(fly_ids)} flies")
    print(f"   Failed: {failed_count}/{len(fly_ids)} flies")

    # Concatenate data and get ball positions
    print("\n7. Creating heatmaps...")
    combo_data = {}
    ball_positions = {}

    for genotype in genotypes:
        for pretraining in pretrainings:
            combo_key = (genotype, pretraining)
            if aligned_data_by_combo[combo_key]:
                combo_df = pd.concat(aligned_data_by_combo[combo_key], ignore_index=True)
                combo_data[combo_key] = combo_df
                n_flies = combo_df["fly"].nunique()
                n_points = len(combo_df)
                print(f"   {genotype} + {pretraining}: {n_flies} flies, {n_points:,} points")

                # Get ball position
                sample_fly = combo_df["fly"].iloc[0]
                sample_video = get_video_path_from_fly_id(sample_fly, None)
                if sample_video:
                    sample_arena_params = detect_arena_in_video(sample_video, template_binary)
                    if sample_arena_params:
                        ball_pos = get_ball_position_in_template(
                            combo_df, sample_arena_params, template.shape, pretraining
                        )
                        if ball_pos:
                            ball_positions[combo_key] = ball_pos
                            print(f"     Ball position: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")

    # Create figure with 2x2 subplots (Pretraining x Genotype)
    n_combos = len(combo_data)
    if n_combos == 0:
        print("\n   ERROR: No data to plot!")
        return

    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # Pre-compute all heatmaps
    print("\n   Computing heatmaps...")
    heatmaps = {}
    global_vmax = 0

    for combo_key, data in combo_data.items():
        heatmap = create_heatmap_on_template(
            data, template, template_binary, bins=args.bins, blur_sigma=args.blur_sigma
        )
        heatmaps[combo_key] = heatmap
        if heatmap is not None:
            combo_max = np.nanmax(heatmap)
            if combo_max > global_vmax:
                global_vmax = combo_max

    print(f"   Global vmax for consistent scaling: {global_vmax:.6f}")

    # Create scale configurations
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

    # Test mode early exit
    if args.test_mode:
        print("\nTEST MODE: Plotting debug heatmap...")
        fig, axes = plt.subplots(
            len(genotypes), len(pretrainings), figsize=(8 * len(pretrainings), 10 * len(genotypes))
        )

        for i, genotype in enumerate(genotypes):
            for j, pretraining in enumerate(pretrainings):
                if len(genotypes) == 1 and len(pretrainings) == 1:
                    ax = axes
                elif len(genotypes) == 1:
                    ax = axes[j]
                elif len(pretrainings) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                combo_key = (genotype, pretraining)
                ax.imshow(template_rgb, aspect="auto", alpha=1.0)

                if combo_key in heatmaps and heatmaps[combo_key] is not None:
                    heatmap = heatmaps[combo_key]
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
                    plt.colorbar(im, ax=ax, label="Debug Density", fraction=0.046, pad=0.04)

                ax.set_title(f"DEBUG: {genotype} + {pretraining}", fontsize=14, fontweight="bold")
                ax.set_xlim(0, template.shape[1])
                ax.set_ylim(template.shape[0], 0)
                ax.set_aspect("equal")

        plt.tight_layout()
        debug_path = output_dir / f"debug_heatmap_orientation_{args.mode}.png"
        fig.savefig(debug_path, dpi=200, bbox_inches="tight")
        print(f"   Saved debug heatmap to: {debug_path}")
        plt.close()
        print("\nTest mode complete.")
        return

    # Normal plotting for all scale configs
    for scale_config in scale_configs:
        scale_type = scale_config["name"]
        print(f"\n   Creating {scale_type} scale version...")

        # Calculate vmax for this scale type
        if scale_config["vmax_percentile"] is not None:
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
            scale_vmax = None

        # Create 2x2 figure (genotypes x pretrainings)
        fig, axes = plt.subplots(
            len(genotypes), len(pretrainings), figsize=(8 * len(pretrainings), 10 * len(genotypes))
        )

        for i, genotype in enumerate(genotypes):
            for j, pretraining in enumerate(pretrainings):
                if len(genotypes) == 1 and len(pretrainings) == 1:
                    ax = axes
                elif len(genotypes) == 1:
                    ax = axes[j]
                elif len(pretrainings) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                combo_key = (genotype, pretraining)
                ax.imshow(template_rgb, aspect="auto", alpha=1.0)

                if combo_key in heatmaps and heatmaps[combo_key] is not None:
                    heatmap = heatmaps[combo_key]
                    cmap = plt.get_cmap("viridis")
                    cmap.set_bad(alpha=0)

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
                    )

                    plt.colorbar(im, ax=ax, label=scale_config["label"], fraction=0.046, pad=0.04)

                if combo_key in combo_data:
                    data = combo_data[combo_key]
                    n_flies = data["fly"].nunique()
                    n_points = len(data)
                    ax.set_title(
                        f"{genotype} + {pretraining}\n{n_flies} flies, {n_points:,} points",
                        fontsize=14,
                        fontweight="bold",
                    )
                else:
                    ax.set_title(f"{genotype} + {pretraining}\nNo data", fontsize=14, fontweight="bold")

                ax.set_xlim(0, template.shape[1])
                ax.set_ylim(template.shape[0], 0)
                ax.set_xlabel("X (template pixels)", fontsize=12)
                ax.set_ylabel("Y (template pixels)", fontsize=12)
                ax.set_aspect("equal")

        plt.tight_layout()

        # Save standard version
        mode_suffix = "_pooled" if "pooled" in args.mode else ""
        output_filename = f"aligned_heatmaps_{scale_type}{mode_suffix}.png"
        output_path = output_dir / output_filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"   Saved {scale_type} scale heatmap to: {output_path}")

        plt.close()

        # Create versions with proximity circles
        if ball_positions:
            proximity_radii = [70, 140, 200]

            for proximity_radius_px in proximity_radii:
                print(f"   Creating {scale_type} scale version with {proximity_radius_px}px proximity circle...")
                fig, axes = plt.subplots(
                    len(genotypes), len(pretrainings), figsize=(8 * len(pretrainings), 10 * len(genotypes))
                )

                for i, genotype in enumerate(genotypes):
                    for j, pretraining in enumerate(pretrainings):
                        if len(genotypes) == 1 and len(pretrainings) == 1:
                            ax = axes
                        elif len(genotypes) == 1:
                            ax = axes[j]
                        elif len(pretrainings) == 1:
                            ax = axes[i]
                        else:
                            ax = axes[i, j]

                        combo_key = (genotype, pretraining)
                        ax.imshow(template_rgb, aspect="auto", alpha=1.0)

                        if combo_key in heatmaps and heatmaps[combo_key] is not None:
                            heatmap = heatmaps[combo_key]
                            cmap = plt.get_cmap("viridis")
                            cmap.set_bad(alpha=0)

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
                            )

                            plt.colorbar(im, ax=ax, label=scale_config["label"], fraction=0.046, pad=0.04)

                        # Draw proximity circle if ball position is available
                        if combo_key in ball_positions:
                            ball_x, ball_y = ball_positions[combo_key]

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
                            ax.plot(ball_x, ball_y, "r+", markersize=10, markeredgewidth=2, alpha=0.8)

                        if combo_key in combo_data:
                            data = combo_data[combo_key]
                            n_flies = data["fly"].nunique()
                            n_points = len(data)
                            title = f"{genotype} + {pretraining}\n{n_flies} flies, {n_points:,} points"
                            if combo_key in ball_positions:
                                title += f"\n(red circle: {proximity_radius_px}px proximity)"
                            ax.set_title(title, fontsize=14, fontweight="bold")
                        else:
                            ax.set_title(f"{genotype} + {pretraining}\nNo data", fontsize=14, fontweight="bold")

                        ax.set_xlim(0, template.shape[1])
                        ax.set_ylim(template.shape[0], 0)
                        ax.set_xlabel("X (template pixels)", fontsize=12)
                        ax.set_ylabel("Y (template pixels)", fontsize=12)
                        ax.set_aspect("equal")

                plt.tight_layout()

                output_filename_prox = (
                    f"aligned_heatmaps_{scale_type}_with_proximity_{proximity_radius_px}px{mode_suffix}.png"
                )
                output_path_prox = output_dir / output_filename_prox
                fig.savefig(output_path_prox, dpi=300, bbox_inches="tight")
                print(
                    f"   Saved {scale_type} scale heatmap with {proximity_radius_px}px proximity to: {output_path_prox}"
                )

                plt.close()

    print("\n8. COMPLETE!")
    print("=" * 70)
    print(f"Generated heatmaps for {args.mode}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
