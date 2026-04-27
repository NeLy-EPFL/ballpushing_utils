#!/usr/bin/env python3
"""
Test binary template matching across multiple videos for reproducibility.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


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
    elif method == "threshold":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return binary


def add_padding(image, padding_percent=0.4, is_binary=False):
    """Add padding around image."""
    h, w = image.shape[:2]
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)

    if is_binary or len(image.shape) == 2:
        if len(image.shape) == 2:
            padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0])
        else:
            padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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


def process_video(video_path, template, template_binary, binarization_method="otsu"):
    """Process a single video and return matching results."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Add padding
    frame_padded, padding = add_padding(frame, padding_percent=0.4, is_binary=False)

    # Binarize
    frame_binary = binarize_video_frame(frame_padded, method=binarization_method)

    # Match
    match_result = match_binary_multiscale(frame_binary, template_binary, scale_range=(0.3, 0.8), scale_steps=60)

    if match_result is None:
        return None

    # Adjust location for original frame
    original_loc = (match_result["location"][0] - padding[2], match_result["location"][1] - padding[0])

    return {
        "video_path": str(video_path),
        "score": match_result["score"],
        "scale": match_result["scale"],
        "template_width": match_result["size"][0],
        "template_height": match_result["size"][1],
        "location_x": original_loc[0],
        "location_y": original_loc[1],
        "padded_location_x": match_result["location"][0],
        "padded_location_y": match_result["location"][1],
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "total_frames": total_frames,
    }


def visualize_multiple_matches(results, template, output_path):
    """Create a grid visualization of multiple video matches."""
    n_videos = len(results)
    n_cols = min(5, n_videos)
    n_rows = (n_videos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Load video frame
        cap = cv2.VideoCapture(result["video_path"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Draw rectangle
            loc = (result["location_x"], result["location_y"])
            w, h = result["template_width"], result["template_height"]

            if loc[0] >= 0 and loc[1] >= 0 and loc[0] + w <= frame.shape[1] and loc[1] + h <= frame.shape[0]:
                cv2.rectangle(frame, loc, (loc[0] + w, loc[1] + h), (0, 255, 0), 2)

            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extract video name
            video_name = Path(result["video_path"]).parent.name + "/" + Path(result["video_path"]).name
            ax.set_title(f"{video_name}\nScore: {result['score']:.3f}, Scale: {result['scale']:.3f}", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")

        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_videos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    videos_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked")
    output_dir = Path("/home/durrieu/ballpushing_utils/outputs/template_matching_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template
    print("Loading template...")
    template = cv2.imread(str(template_path))
    template_binary = binarize_template(template, invert_background=True)
    print(f"Template shape: {template.shape}")

    # Find video files (looking in subdirectories)
    print(f"\nSearching for videos in: {videos_dir}")
    video_files = []

    # Look for .mp4 files in subdirectories
    for video_path in videos_dir.rglob("*.mp4"):
        video_files.append(video_path)

    print(f"Found {len(video_files)} video files")

    # Sample up to 10 videos
    if len(video_files) > 10:
        import random

        random.seed(42)
        video_files = random.sample(video_files, 10)
        print(f"Randomly sampled 10 videos for testing")

    # Process videos
    results = []
    print("\nProcessing videos...")
    print("=" * 70)

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.relative_to(videos_dir)}")

        result = process_video(video_path, template, template_binary, binarization_method="otsu")

        if result is not None:
            results.append(result)
            print(f"  Score: {result['score']:.4f}")
            print(f"  Scale: {result['scale']:.3f}")
            print(f"  Template size: {result['template_width']}x{result['template_height']}")
            print(f"  Location: ({result['location_x']}, {result['location_y']})")
        else:
            print(f"  FAILED - No match found")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = output_dir / "multi_video_matching_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved results to: {csv_path}")

        # Print statistics
        print(f"\n{'='*70}")
        print("STATISTICS:")
        print(f"{'='*70}")
        print(f"Videos successfully processed: {len(results)}/{len(video_files)}")
        print(f"\nMatch scores:")
        print(f"  Mean:   {df['score'].mean():.4f}")
        print(f"  Std:    {df['score'].std():.4f}")
        print(f"  Min:    {df['score'].min():.4f}")
        print(f"  Max:    {df['score'].max():.4f}")
        print(f"\nScale factors:")
        print(f"  Mean:   {df['scale'].mean():.4f}")
        print(f"  Std:    {df['scale'].std():.4f}")
        print(f"  Min:    {df['scale'].min():.4f}")
        print(f"  Max:    {df['scale'].max():.4f}")
        print(f"\nTemplate sizes:")
        print(f"  Width  - Mean: {df['template_width'].mean():.1f}, Std: {df['template_width'].std():.1f}")
        print(f"  Height - Mean: {df['template_height'].mean():.1f}, Std: {df['template_height'].std():.1f}")

        # Create visualization
        print(f"\n{'='*70}")
        print("Creating visualization...")
        viz_path = output_dir / "multi_video_matching_visualization.png"
        visualize_multiple_matches(results, template, viz_path)

        print(f"\n{'='*70}")
        print("DONE!")
        print(f"Results CSV: {csv_path}")
        print(f"Visualization: {viz_path}")
    else:
        print("\nNo videos were successfully processed!")


if __name__ == "__main__":
    main()
