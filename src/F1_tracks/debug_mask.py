#!/usr/bin/env python3
"""Debug script to visualize the mask downsampling."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def binarize_template(template, invert_background=True):
    """Binarize the template image."""
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if invert_background:
        return binary
    else:
        return cv2.bitwise_not(binary)


template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
template = cv2.imread(str(template_path))
template_binary = binarize_template(template, invert_background=True)

bins = 100

# Method 1: INTER_NEAREST (original)
mask_nearest = cv2.resize(template_binary.astype(np.uint8), (bins, bins), interpolation=cv2.INTER_NEAREST)
mask_nearest_bool = mask_nearest > 0

# Method 2: INTER_LINEAR with threshold 127
mask_linear = cv2.resize(template_binary.astype(np.uint8), (bins, bins), interpolation=cv2.INTER_LINEAR)
mask_linear_bool = mask_linear > 127

# Method 3: INTER_LINEAR with threshold 50 (more lenient)
mask_linear_50 = mask_linear > 50

# Method 4: INTER_LINEAR with threshold 127 + dilation
mask_linear_dilated = mask_linear > 127
kernel = np.ones((3, 3), np.uint8)
mask_linear_dilated = cv2.dilate(mask_linear_dilated.astype(np.uint8), kernel, iterations=1).astype(bool)

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Show original template
axes[0, 0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Original Template\n{template.shape[1]}x{template.shape[0]} pixels", fontsize=12)
axes[0, 0].axis("off")

# Show binary template
axes[0, 1].imshow(template_binary, cmap="gray")
axes[0, 1].set_title(
    f"Binary Template (threshold=200)\n{template_binary.shape[1]}x{template_binary.shape[0]} pixels", fontsize=12
)
axes[0, 1].axis("off")

# Show linear interpolated (before threshold)
axes[0, 2].imshow(mask_linear, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title(f"INTER_LINEAR downscaled\n{bins}x{bins} bins (before threshold)", fontsize=12)
axes[0, 2].axis("off")

# Method 1
axes[1, 0].imshow(mask_nearest_bool, cmap="gray")
white_pixels_1 = np.sum(mask_nearest_bool)
axes[1, 0].set_title(f"INTER_NEAREST > 0\n{white_pixels_1}/{bins*bins} pixels white", fontsize=12)
axes[1, 0].axis("off")

# Method 2
axes[1, 1].imshow(mask_linear_bool, cmap="gray")
white_pixels_2 = np.sum(mask_linear_bool)
axes[1, 1].set_title(f"INTER_LINEAR > 127\n{white_pixels_2}/{bins*bins} pixels white", fontsize=12)
axes[1, 1].axis("off")

# Method 4
axes[1, 2].imshow(mask_linear_dilated, cmap="gray")
white_pixels_4 = np.sum(mask_linear_dilated)
axes[1, 2].set_title(f"INTER_LINEAR > 127 + dilate(3)\n{white_pixels_4}/{bins*bins} pixels white", fontsize=12)
axes[1, 2].axis("off")

plt.tight_layout()

output_path = Path(__file__).parent / "mask_downsampling_comparison.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved comparison to: {output_path}")

# Also save individual masks
cv2.imwrite(str(Path(__file__).parent / "mask_nearest.png"), mask_nearest_bool.astype(np.uint8) * 255)
cv2.imwrite(str(Path(__file__).parent / "mask_linear.png"), mask_linear_bool.astype(np.uint8) * 255)
cv2.imwrite(str(Path(__file__).parent / "mask_linear_dilated.png"), mask_linear_dilated.astype(np.uint8) * 255)

print(f"Template size: {template.shape}")
print(f"Downsampling ratio: {template.shape[0]/bins:.1f}x")
print(f"\nWhite pixel counts:")
print(f"  INTER_NEAREST > 0: {white_pixels_1}/{bins*bins} ({100*white_pixels_1/(bins*bins):.1f}%)")
print(f"  INTER_LINEAR > 127: {white_pixels_2}/{bins*bins} ({100*white_pixels_2/(bins*bins):.1f}%)")
print(f"  INTER_LINEAR > 127 + dilate(3): {white_pixels_4}/{bins*bins} ({100*white_pixels_4/(bins*bins):.1f}%)")
