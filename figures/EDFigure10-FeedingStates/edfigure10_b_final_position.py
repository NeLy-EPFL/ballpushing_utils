#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 10b: Distribution of final ball positions
by nutritional state.

This script generates a 3-row × 1-column figure (one subplot per nutritional state)
showing:
- Final ball position relative to start (mm) per fly
- Light ON condition only (Dark experiments excluded)
- Kernel density estimate overlaid
- Median value as a dashed vertical line
- Color-coded by FeedingState (fed=green, starved=orange, starved_noWater=blue)

Usage:
    python edfigure10_b_final_position.py [--bins N] [--output-dir PATH]

Arguments:
    --bins:        Number of histogram bins (default: 25)
    --output-dir:  Override output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PX_PER_MM = 500 / 30

FEEDING_ORDER = ["fed", "starved", "starved_noWater"]
FEEDING_LABELS = {
    "fed": "Fed",
    "starved": "Starved",
    "starved_noWater": "Starved (no water)",
}
FEEDING_COLORS = {
    "fed": "#66C2A5",  # green
    "starved": "#FC8D62",  # orange
    "starved_noWater": "#8DA0CB",  # blue
}

DEFAULT_COORDINATES_DIR = dataset("Ballpushing_Exploration/Datasets/260220_10_summary_control_folders_Data/coordinates")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_final_positions(coordinates_dir: Path) -> pd.DataFrame:
    """Load all coordinate feather files and compute per-fly final ball position.

    Skips files with 'dark' in the filename and rows where Light != 'on'.

    Returns a DataFrame with columns: fly, FeedingState, final_position_mm
    """
    feather_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in {coordinates_dir}")

    non_dark = [f for f in feather_files if "dark" not in f.name.lower()]
    print(f"Files found: {len(feather_files)}  |  Dark excluded: {len(feather_files) - len(non_dark)}")

    chunks = []
    for i, fp in enumerate(non_dark, 1):
        print(f"  [{i}/{len(non_dark)}] {fp.name}")
        df = read_feather(fp)

        # Keep Light=on only
        if "Light" in df.columns:
            df = df[df["Light"].astype(str) == "on"]

        if df.empty:
            continue

        # Normalise FeedingState
        if "FeedingState" not in df.columns:
            continue
        rename_map = {
            "Fed": "fed",
            "fed": "fed",
            "starved": "starved",
            "starved_noWater": "starved_noWater",
        }
        df = df.copy()
        df["FeedingState"] = df["FeedingState"].astype(str).str.strip().map(rename_map).fillna(df["FeedingState"])
        df = df[df["FeedingState"].isin(FEEDING_ORDER)]
        if df.empty:
            continue

        # Per-fly: first and last distance_ball_0 (sorted by time)
        sorted_df = df.sort_values("time")
        first_dist = sorted_df.groupby("fly")["distance_ball_0"].first()
        last_dist = sorted_df.groupby("fly")["distance_ball_0"].last()
        feeding = sorted_df.groupby("fly")["FeedingState"].first().astype(str)

        fly_df = pd.DataFrame(
            {
                "fly": first_dist.index,
                "FeedingState": feeding.values,
                "final_position_mm": (last_dist.values - first_dist.values) / PX_PER_MM,
            }
        )
        # Prefix fly IDs with file stem to avoid collisions
        fly_df["fly"] = fp.stem + "::" + fly_df["fly"].astype(str)
        chunks.append(fly_df)

    if not chunks:
        raise ValueError("No data loaded.")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal flies: {len(combined)}")
    for fs in FEEDING_ORDER:
        n = (combined["FeedingState"] == fs).sum()
        if n:
            print(f"  {fs}: {n} flies")
    return combined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_final_position_subplots(
    data: pd.DataFrame,
    n_bins: int = 25,
    output_path: Path | None = None,
) -> None:
    """3-row × 1-column figure — one subplot per nutritional state.

    Each panel shows: histogram (density) + KDE curve + dashed median line.
    All subplots share the same x-axis so positions are directly comparable.
    """
    conditions = [c for c in FEEDING_ORDER if c in data["FeedingState"].values]

    # Shared x range across all conditions
    x_min = data["final_position_mm"].min()
    x_max = data["final_position_mm"].max()
    x_pad = (x_max - x_min) * 0.05
    x_lim = (x_min - x_pad, x_max + x_pad)
    bin_edges = np.linspace(x_lim[0], x_lim[1], n_bins + 1)
    xs = np.linspace(x_lim[0], x_lim[1], 500)

    n_rows = len(conditions)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(100 / 25.4, n_rows * 70 / 25.4),
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        vals = data.loc[data["FeedingState"] == cond, "final_position_mm"].dropna()
        color = FEEDING_COLORS.get(cond, "gray")
        label = FEEDING_LABELS.get(cond, cond)
        n_flies = len(vals)

        ax.hist(
            vals,
            bins=bin_edges,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.7,
            density=True,
        )

        if n_flies > 1:
            kde_curve = gaussian_kde(vals, bw_method="scott")
            ax.plot(xs, kde_curve(xs), color=color, linewidth=2.0)

        med = vals.median()
        ax.axvline(
            med,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )

        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{label} (n={n_flies})", fontsize=10, color=color, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_xlim(x_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Final ball position (mm)", fontsize=10)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path.with_suffix('.pdf')}")
        print(f"Saved: {output_path.with_suffix('.png')}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extended Data Figure 10b: final ball position by nutritional state")
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=DEFAULT_COORDINATES_DIR,
        help=f"Directory of coordinates feather files (default: {DEFAULT_COORDINATES_DIR})",
    )
    parser.add_argument("--bins", type=int, default=25, help="Number of histogram bins (default: 25)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else figure_output_dir("EDFigure10", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 10b: Final ball position by nutritional state")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    data = load_final_positions(args.coordinates_dir)

    output_path = output_dir / "edfigure10b_final_position"
    plot_final_position_subplots(data, n_bins=args.bins, output_path=output_path)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
