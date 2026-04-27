#!/usr/bin/env python3
"""
Test script for template matching to detect arena position in videos.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_template(template_path):
    """Load the template image."""
    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Could not load template from {template_path}")
    return template


def extract_video_frame(video_path, frame_number=-1):
    """
    Extract a frame from video.

    Args:
        video_path: Path to video file
        frame_number: Frame to extract. If -1, gets the last frame.

    Returns:
        numpy array of the frame
    """
    cap = cv2.VideoCapture(str(video_path))

    if frame_number == -1:
        # Get last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = max(0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")

    return frame


def find_arena_multiscale_matching(
    frame, template, method=cv2.TM_CCOEFF_NORMED, scale_range=(0.5, 1.2), scale_steps=30
):
    """
    Find arena in frame using multi-scale template matching.

    Args:
        frame: Video frame (BGR)
        template: Template image (BGR)
        method: OpenCV template matching method
        scale_range: Tuple of (min_scale, max_scale) to search
        scale_steps: Number of scales to try

    Returns:
        tuple: (top_left_x, top_left_y, match_score, resized_template, scale_factor)
    """
    frame_h, frame_w = frame.shape[:2]
    template_h, template_w = template.shape[:2]

    best_score = -np.inf if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else np.inf
    best_loc = None
    best_scale = None
    best_template = None

    # Convert frame to grayscale once
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    print(f"Searching scales from {scale_range[0]:.2f} to {scale_range[1]:.2f}...")

    for scale in scales:
        # Resize template
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        # Skip if template is larger than frame
        if new_w > frame_w or new_h > frame_h:
            continue

        resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        template_gray = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)

        # Apply template matching
        result = cv2.matchTemplate(frame_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Get score and location based on method
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            score = min_val
            loc = min_loc
            is_better = score < best_score
        else:
            score = max_val
            loc = max_loc
            is_better = score > best_score

        # Update best match
        if is_better:
            best_score = score
            best_loc = loc
            best_scale = scale
            best_template = resized_template

    if best_loc is None:
        # Fallback to single scale
        scale = min(frame_w / template_w, frame_h / template_h)
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)
        best_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        best_scale = scale
        best_loc = (0, 0)
        best_score = 0.0

    print(
        f"Best match at scale {best_scale:.3f}: template size {best_template.shape[1]}x{best_template.shape[0]}, score: {best_score:.4f}"
    )

    return best_loc[0], best_loc[1], best_score, best_template, best_scale


def visualize_detection(frame, template, top_left_x, top_left_y, match_score):
    """
    Visualize the detected arena position.

    Args:
        frame: Original video frame
        template: Template image
        top_left_x, top_left_y: Detected position
        match_score: Matching score

    Returns:
        Figure with visualization
    """
    h, w = template.shape[:2]
    bottom_right_x = top_left_x + w
    bottom_right_y = top_left_y + h

    # Draw rectangle on frame
    frame_with_rect = frame.copy()
    cv2.rectangle(frame_with_rect, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)

    # Create overlay
    overlay = frame.copy()
    overlay[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = cv2.addWeighted(
        overlay[top_left_y:bottom_right_y, top_left_x:bottom_right_x], 0.5, template, 0.5, 0
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Frame")
    axes[0, 0].axis("off")

    # Template
    axes[0, 1].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Template")
    axes[0, 1].axis("off")

    # Detection
    axes[1, 0].imshow(cv2.cvtColor(frame_with_rect, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Detected Arena (Score: {match_score:.3f})")
    axes[1, 0].axis("off")

    # Overlay
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Template Overlay")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def extract_arena_mask(template):
    """
    Extract the arena mask from template (white = arena, gray = background).

    Args:
        template: Template image (BGR)

    Returns:
        Binary mask (1 = arena, 0 = background)
    """
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Threshold to separate white arena from gray background
    # Assuming white > 200 and gray < 200
    _, mask = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY)

    return mask


def main():
    """Test template matching on sample video."""

    # Paths
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    video_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked/arena6/Left/Left.mp4")
    output_dir = Path("/home/durrieu/ballpushing_utils/outputs/template_matching_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading template from: {template_path}")
    template = load_template(template_path)
    print(f"Template shape: {template.shape}")

    print(f"\nLoading video from: {video_path}")

    # Test on multiple frames
    test_frames = [0, 100, -1]  # First, middle-ish, and last frame

    # Test different matching methods
    methods = [
        (cv2.TM_CCOEFF_NORMED, "CCOEFF_NORMED"),
        (cv2.TM_CCORR_NORMED, "CCORR_NORMED"),
        (cv2.TM_SQDIFF_NORMED, "SQDIFF_NORMED"),
    ]

    for method, method_name in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method_name}")
        print(f"{'='*60}")

        for frame_idx in test_frames:
            print(f"\n--- Testing frame {frame_idx} ---")
            frame = extract_video_frame(video_path, frame_idx)
            print(f"Frame shape: {frame.shape}")

            # Perform template matching
            top_left_x, top_left_y, match_score, resized_template, scale_factor = find_arena_multiscale_matching(
                frame, template, method=method, scale_range=(0.5, 1.2), scale_steps=30
            )

            print(f"Detected position: ({top_left_x}, {top_left_y})")
            print(f"Match score: {match_score:.4f}")

            # Visualize
            fig = visualize_detection(frame, resized_template, top_left_x, top_left_y, match_score)

            # Save
            frame_label = "last" if frame_idx == -1 else f"frame{frame_idx:04d}"
            output_path = output_dir / f"template_match_{method_name}_{frame_label}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to: {output_path}")
            plt.close(fig)

    # Extract and visualize the arena mask
    print("\n--- Extracting arena mask ---")
    mask = extract_arena_mask(template)

    fig_mask, axes_mask = plt.subplots(1, 2, figsize=(12, 6))
    axes_mask[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes_mask[0].set_title("Template Image")
    axes_mask[0].axis("off")

    axes_mask[1].imshow(mask, cmap="gray")
    axes_mask[1].set_title("Arena Mask (White = Arena)")
    axes_mask[1].axis("off")

    plt.tight_layout()
    mask_output_path = output_dir / "arena_mask.png"
    fig_mask.savefig(mask_output_path, dpi=150, bbox_inches="tight")
    print(f"Saved arena mask to: {mask_output_path}")
    plt.close(fig_mask)

    print("\n=== Template matching test complete! ===")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
