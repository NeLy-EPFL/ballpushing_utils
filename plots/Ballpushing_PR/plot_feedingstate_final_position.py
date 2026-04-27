#!/usr/bin/env python3
"""
Histogram of final ball position (displacement from start, in mm) for each fly,
grouped by FeedingState.

One subplot per FeedingState condition (fed / starved / starved_noWater),
using the canonical colour palette.  Light=on only; Dark experiments excluded.

Usage:
    python plot_feedingstate_final_position.py
    python plot_feedingstate_final_position.py --coordinates-dir PATH --output-dir PATH
    python plot_feedingstate_final_position.py --bins 30 --no-kde
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

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
    "fed": "#66C2A5",
    "starved": "#FC8D62",
    "starved_noWater": "#8DA0CB",
}

DEFAULT_COORDINATES_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets" "/260220_10_summary_control_folders_Data/coordinates"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/feedingstate_final_position")


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
        df = pd.read_feather(fp)

        # Keep Light=on only
        if "Light" in df.columns:
            df = df[df["Light"].astype(str) == "on"]

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


def plot_final_position_histograms(
    data: pd.DataFrame,
    n_bins: int = 25,
    kde: bool = True,
    output_path: Path | None = None,
) -> None:
    """One subplot per FeedingState condition with histogram (+ optional KDE)."""
    conditions = [c for c in FEEDING_ORDER if c in data["FeedingState"].values]
    n_conds = len(conditions)

    # Shared x range across all subplots
    x_min = data["final_position_mm"].min()
    x_max = data["final_position_mm"].max()
    x_pad = (x_max - x_min) * 0.05
    x_lim = (x_min - x_pad, x_max + x_pad)

    fig, axes = plt.subplots(
        n_conds,
        1,
        figsize=(140 / 25.4, 250 / 25.4),
        sharey=False,
        sharex=True,
    )
    if n_conds == 1:
        axes = [axes]

    bin_edges = np.linspace(x_min - x_pad, x_max + x_pad, n_bins + 1)

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
            alpha=0.85,
            density=kde,  # density=True when KDE is shown, count otherwise
        )

        if kde and n_flies > 1:
            from scipy.stats import gaussian_kde

            xs = np.linspace(x_lim[0], x_lim[1], 400)
            kde_curve = gaussian_kde(vals, bw_method="scott")
            ax.plot(xs, kde_curve(xs), color=color, linewidth=2)

        # Median line
        med = vals.median()
        ax.axvline(med, color="black", linewidth=1, linestyle="--", alpha=0.7, label=f"Median: {med:.1f} mm")

        ax.set_title(f"{label}\n(n={n_flies})", fontsize=12)
        ax.set_ylabel("Density" if kde else "Count", fontsize=12)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, framealpha=0.6)
        ax.set_xlim(x_lim)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Final ball position (mm)", fontsize=12)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        print(f"Saved: {output_path.with_suffix('.svg')}")
        print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Histogram of final ball position by FeedingState")
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=DEFAULT_COORDINATES_DIR,
        help=f"Directory of coordinates feather files (default: {DEFAULT_COORDINATES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of histogram bins (default: 25)",
    )
    parser.add_argument(
        "--no-kde",
        action="store_true",
        help="Show counts instead of density + KDE curve",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="feedingstate_final_position",
        help="Base filename for output plots (default: feedingstate_final_position)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from: {args.coordinates_dir}")
    data = load_final_positions(args.coordinates_dir)

    output_path = args.output_dir / args.base_name
    plot_final_position_histograms(
        data,
        n_bins=args.bins,
        kde=not args.no_kde,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
