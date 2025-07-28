#!/usr/bin/env python3
"""
Create separate legend figures for PCA scatterplot matrix using matplotlib/seaborn
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config


def create_brain_region_legend():
    """Create brain region color legend"""
    brain_region_colors = Config.color_dict

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter points for each brain region
    y_positions = np.arange(len(brain_region_colors))

    for i, (region, color) in enumerate(brain_region_colors.items()):
        ax.scatter(0, i, c=color, s=100, alpha=0.8, edgecolors="black", linewidth=1)
        ax.text(0.1, i, region, fontsize=12, va="center")

    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.5, len(brain_region_colors) - 0.5)
    ax.set_title("Brain Regions Color Code", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("pca_legend_brain_regions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Brain region legend saved as 'pca_legend_brain_regions.png'")


def create_significance_legend():
    """Create significance types (marker shapes) legend"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Define markers and labels
    markers_data = [
        ("circle", "o", "Both Static & Temporal", 12),
        ("square", "s", "Static Only", 10),
        ("triangle", "^", "Temporal Only", 10),
    ]

    y_positions = np.arange(len(markers_data))

    for i, (shape_name, marker, label, size) in enumerate(markers_data):
        ax.scatter(0, i, marker=marker, s=size * 10, c="gray", alpha=0.8, edgecolors="black", linewidth=1)
        ax.text(0.1, i, label, fontsize=12, va="center")

    ax.set_xlim(-0.2, 2.5)
    ax.set_ylim(-0.5, len(markers_data) - 0.5)
    ax.set_title("Significance Types (Marker Shapes)", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("pca_legend_significance_types.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Significance types legend saved as 'pca_legend_significance_types.png'")


def create_control_ellipses_legend():
    """Create control ellipses color legend"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Define control lines and colors
    controls_data = [("Empty-Split", "#ffb7b7"), ("Empty-Gal4", "#C6F7B0"), ("TNTxPR", "#B4EAFF")]

    y_positions = np.arange(len(controls_data))

    for i, (control_name, color) in enumerate(controls_data):
        # Draw a dashed line
        ax.plot([0, 0.3], [i, i], color=color, linewidth=4, linestyle="--", alpha=0.8)
        ax.text(0.4, i, f"{control_name} (95% CI)", fontsize=12, va="center")

    ax.set_xlim(-0.1, 2.5)
    ax.set_ylim(-0.5, len(controls_data) - 0.5)
    ax.set_title("Control Ellipses Color Code", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("pca_legend_control_ellipses.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Control ellipses legend saved as 'pca_legend_control_ellipses.png'")


def create_combined_legend():
    """Create a single combined legend figure with all three legend types"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # 1. Brain regions
    brain_region_colors = Config.color_dict
    for i, (region, color) in enumerate(brain_region_colors.items()):
        ax1.scatter(0, i, c=color, s=80, alpha=0.8, edgecolors="black", linewidth=1)
        ax1.text(0.1, i, region, fontsize=10, va="center")

    ax1.set_xlim(-0.2, 1.5)
    ax1.set_ylim(-0.5, len(brain_region_colors) - 0.5)
    ax1.set_title("Brain Regions\n(Colors)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Significance types
    markers_data = [("o", "Both Static & Temporal", 12), ("s", "Static Only", 10), ("^", "Temporal Only", 10)]

    for i, (marker, label, size) in enumerate(markers_data):
        ax2.scatter(0, i, marker=marker, s=size * 8, c="gray", alpha=0.8, edgecolors="black", linewidth=1)
        ax2.text(0.1, i, label, fontsize=10, va="center")

    ax2.set_xlim(-0.2, 2.2)
    ax2.set_ylim(-0.5, len(markers_data) - 0.5)
    ax2.set_title("Significance Types\n(Marker Shapes)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Control ellipses
    controls_data = [("Empty-Split", "#ffb7b7"), ("Empty-Gal4", "#C6F7B0"), ("TNTxPR", "#B4EAFF")]

    for i, (control_name, color) in enumerate(controls_data):
        ax3.plot([0, 0.3], [i, i], color=color, linewidth=4, linestyle="--", alpha=0.8)
        ax3.text(0.4, i, f"{control_name}", fontsize=10, va="center")

    ax3.set_xlim(-0.1, 1.8)
    ax3.set_ylim(-0.5, len(controls_data) - 0.5)
    ax3.set_title("Control Ellipses\n(95% CI)", fontsize=12, fontweight="bold")
    ax3.axis("off")

    plt.suptitle("PCA Scatterplot Matrix Legend", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Ensure pca_matrices directory exists
    os.makedirs("pca_matrices", exist_ok=True)

    # Save legend to pca_matrices directory
    plt.savefig("pca_matrices/pca_legend_combined.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Combined legend saved as 'pca_matrices/pca_legend_combined.png'")


def main():
    """Create combined legend figure only"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    print("Creating PCA combined legend figure...")

    # Create only the combined legend
    create_combined_legend()

    print("\nLegend file created:")
    print("- pca_matrices/pca_legend_combined.png")


if __name__ == "__main__":
    main()
