#!/usr/bin/env python3
"""
Arena alignment utilities for standardizing fly coordinates to template space.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict


def binarize_template(template, invert_background=True):
    """
    Binarize the template image.

    Args:
        template: Template image (BGR)
        invert_background: If True, make background black (arena white)

    Returns:
        Binary template (white=255 for arena, black=0 for background)
    """
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if invert_background:
        return binary
    else:
        return cv2.bitwise_not(binary)


def binarize_video_frame(frame, method="otsu"):
    """
    Binarize video frame to match template.

    Args:
        frame: Video frame (BGR)
        method: 'otsu' or 'threshold'

    Returns:
        Binary frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return binary


def add_padding(image, padding_percent=0.4):
    """
    Add padding around image to allow for larger template matching.

    Args:
        image: Input image
        padding_percent: Percentage of image dimensions to add as padding

    Returns:
        Padded image and padding amounts (top, bottom, left, right)
    """
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
    """
    Match binary template to binary frame at multiple scales.

    Args:
        frame_binary: Binarized video frame
        template_binary: Binarized template
        scale_range: Range of scales to search
        scale_steps: Number of scales to try

    Returns:
        Dictionary with match results or None
    """
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
        "template_size": (best_template.shape[1], best_template.shape[0]),
    }


def detect_arena_in_video(
    video_path: Path, template: np.ndarray, template_binary: np.ndarray, padding_percent: float = 0.4
) -> Optional[Dict]:
    """
    Detect arena position and scale in a video using the last frame.

    Args:
        video_path: Path to video file
        template: Original template image (BGR)
        template_binary: Binary template
        padding_percent: Padding to add for matching

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

    # Calculate transformation parameters
    # Location in padded frame
    padded_loc = match_result["location"]

    # Location in original frame (accounting for padding)
    original_loc = (padded_loc[0] - padding[2], padded_loc[1] - padding[0])

    # Template size in video coordinates
    template_w, template_h = match_result["template_size"]

    return {
        "video_path": str(video_path),
        "score": match_result["score"],
        "scale": match_result["scale"],
        "arena_x": original_loc[0],  # Top-left corner in original frame
        "arena_y": original_loc[1],
        "arena_width": template_w,
        "arena_height": template_h,
        "template_width": template.shape[1],  # Original template size
        "template_height": template.shape[0],
        "video_width": frame.shape[1],
        "video_height": frame.shape[0],
    }


def transform_coordinates_to_template(
    x_video: np.ndarray, y_video: np.ndarray, arena_params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform fly coordinates from video space to template space.

    Args:
        x_video: X coordinates in video frame
        y_video: Y coordinates in video frame
        arena_params: Arena detection parameters from detect_arena_in_video

    Returns:
        Tuple of (x_template, y_template) coordinates
    """
    # Translate to arena-relative coordinates
    x_arena = x_video - arena_params["arena_x"]
    y_arena = y_video - arena_params["arena_y"]

    # Scale to template size
    scale = arena_params["scale"]
    x_template = x_arena / scale
    y_template = y_arena / scale

    return x_template, y_template


def transform_dataframe_to_template(
    df: pd.DataFrame, arena_params: Dict, x_col: str = "x_thorax_fly_0", y_col: str = "y_thorax_fly_0"
) -> pd.DataFrame:
    """
    Transform all fly position coordinates in a dataframe to template space.

    Args:
        df: DataFrame with fly positions
        arena_params: Arena detection parameters
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates

    Returns:
        DataFrame with added template coordinate columns
    """
    df_transformed = df.copy()

    # Transform coordinates
    x_template, y_template = transform_coordinates_to_template(df[x_col].values, df[y_col].values, arena_params)

    # Add new columns
    df_transformed["x_template"] = x_template
    df_transformed["y_template"] = y_template

    return df_transformed


def get_arena_mask(template_binary: np.ndarray) -> np.ndarray:
    """
    Extract the arena mask from binary template.

    Args:
        template_binary: Binary template (255=arena, 0=background)

    Returns:
        Binary mask (True=arena, False=background)
    """
    return template_binary > 0


def load_template(template_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load template and create binary version and mask.

    Args:
        template_path: Path to template PNG

    Returns:
        Tuple of (template_bgr, template_binary, arena_mask)
    """
    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Could not load template from {template_path}")

    template_binary = binarize_template(template, invert_background=True)
    arena_mask = get_arena_mask(template_binary)

    return template, template_binary, arena_mask


if __name__ == "__main__":
    # Test the functions
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    video_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked/arena6/Left/Left.mp4")

    print("Loading template...")
    template, template_binary, arena_mask = load_template(template_path)
    print(f"Template shape: {template.shape}")
    print(f"Arena mask: {arena_mask.sum()} arena pixels out of {arena_mask.size} total")

    print("\nDetecting arena in video...")
    arena_params = detect_arena_in_video(video_path, template, template_binary)

    if arena_params:
        print(f"Detection successful!")
        print(f"  Score: {arena_params['score']:.4f}")
        print(f"  Scale: {arena_params['scale']:.4f}")
        print(f"  Arena position: ({arena_params['arena_x']}, {arena_params['arena_y']})")
        print(f"  Arena size: {arena_params['arena_width']}x{arena_params['arena_height']}")

        # Test coordinate transformation
        test_x = np.array([100, 150, 200])
        test_y = np.array([200, 250, 300])

        x_template, y_template = transform_coordinates_to_template(test_x, test_y, arena_params)

        print(f"\nCoordinate transformation test:")
        print(f"  Video coords: x={test_x}, y={test_y}")
        print(f"  Template coords: x={x_template}, y={y_template}")
    else:
        print("Detection failed!")
