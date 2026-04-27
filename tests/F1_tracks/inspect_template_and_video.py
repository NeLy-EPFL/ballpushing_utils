#!/usr/bin/env python3
"""
Diagnostic script to inspect template and video dimensions/content.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    video_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked/arena6/Left/Left.mp4")
    output_dir = Path("/home/durrieu/ballpushing_utils/outputs/template_matching_tests")

    # Load template
    template = cv2.imread(str(template_path))
    print(f"Template shape: {template.shape}")  # (height, width, channels)
    print(f"Template size: {template.shape[1]}x{template.shape[0]} (WxH)")

    # Load video frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    print(f"\nVideo frame shape: {frame.shape}")
    print(f"Video frame size: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    # Calculate aspect ratios
    template_aspect = template.shape[1] / template.shape[0]  # width / height
    frame_aspect = frame.shape[1] / frame.shape[0]

    print(f"\nTemplate aspect ratio: {template_aspect:.3f}")
    print(f"Video aspect ratio: {frame_aspect:.3f}")

    # Show both images side by side
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Template - original
    axes[0, 0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Template ({template.shape[1]}x{template.shape[0]})")
    axes[0, 0].axis("off")

    # Template - grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(template_gray, cmap="gray")
    axes[0, 1].set_title("Template (Grayscale)")
    axes[0, 1].axis("off")

    # Template - edges
    template_edges = cv2.Canny(template_gray, 50, 150)
    axes[0, 2].imshow(template_edges, cmap="gray")
    axes[0, 2].set_title("Template (Edges)")
    axes[0, 2].axis("off")

    # Video frame - original
    axes[1, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Video Frame ({frame.shape[1]}x{frame.shape[0]})")
    axes[1, 0].axis("off")

    # Video frame - grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    axes[1, 1].imshow(frame_gray, cmap="gray")
    axes[1, 1].set_title("Video Frame (Grayscale)")
    axes[1, 1].axis("off")

    # Video frame - edges
    frame_edges = cv2.Canny(frame_gray, 50, 150)
    axes[1, 2].imshow(frame_edges, cmap="gray")
    axes[1, 2].set_title("Video Frame (Edges)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    output_path = output_dir / "template_video_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to: {output_path}")

    # Try different scale factors manually
    print("\n" + "=" * 60)
    print("Testing different manual scales:")
    print("=" * 60)

    for scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        new_w = int(template.shape[1] * scale)
        new_h = int(template.shape[0] * scale)

        if new_w > frame.shape[1] or new_h > frame.shape[0]:
            print(f"Scale {scale:.1f}: {new_w}x{new_h} - TOO LARGE")
            continue

        print(f"Scale {scale:.1f}: {new_w}x{new_h} - OK")

    plt.show()


if __name__ == "__main__":
    main()
