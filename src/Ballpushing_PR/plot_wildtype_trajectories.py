#!/usr/bin/env python3
"""
Generate BallTrajectories_2-style plots from a coordinates feather file.

Two modes:
  single (default)
      Load one feather file and plot all fly trajectories it contains.
  sample
      Load all feather files from a coordinates directory, apply column
      filters (e.g. FeedingState=starved_noWater Light=on), then randomly
      sample N flies across the matching pool.

Plots produced:
  1) Aligned trajectories – gray background, representative in black,
     interaction onsets in red.
  2) Raw trajectories – gray background, representative segmented red/black
     by interaction state, onsets in red.
  3) Aligned trajectories colored by a grouping column (e.g. FeedingState).

Usage examples:
  python plot_wildtype_trajectories.py
  python plot_wildtype_trajectories.py path/to/file.feather
  python plot_wildtype_trajectories.py --mode sample \\
      --filter FeedingState=starved_noWater --filter Light=on --n-flies 53
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PX_PER_MM = 500 / 30

# Colours matching plot_feedingstate_trajectories.py
FEEDING_COLORS: dict[str, str] = {
    "fed": "#66C2A5",  # teal
    "starved": "#FC8D62",  # orange
    "starved_noWater": "#8DA0CB",  # blue
}

DEFAULT_FEATHER = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets"
    "/260220_10_summary_control_folders_Data/coordinates"
    "/230704_FeedingState_1_AM_Videos_Tracked_coordinates.feather"
)
DEFAULT_COORDINATES_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets" "/260220_10_summary_control_folders_Data/coordinates"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/wildtype_trajectories")


def load_dataset(feather_path: Path) -> pd.DataFrame:
    """Load a coordinates feather file and add derived distance columns."""
    data = pd.read_feather(feather_path)

    if data.empty:
        raise ValueError(f"Dataset is empty: {feather_path}")

    data["distance_ball_0_mm"] = data["distance_ball_0"] / PX_PER_MM
    data["distance_ball_0_mm_aligned"] = data.groupby("fly")["distance_ball_0_mm"].transform(
        lambda values: values - values.iloc[0]
    )

    return data


def load_sampled_dataset(
    coordinates_dir: Path,
    filters: dict[str, str],
    n_flies: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Load feather files from a directory, filter by column values, sample N flies.

    Parameters
    ----------
    coordinates_dir:
        Directory containing ``*_coordinates.feather`` files.
    filters:
        Mapping of column name → expected value, e.g.
        ``{"FeedingState": "starved_noWater", "Light": "on"}``.
        Values are matched as strings (case-sensitive).
    n_flies:
        Number of flies to sample from the filtered pool.
    seed:
        Random seed for reproducibility.
    """
    feather_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in {coordinates_dir}")

    print(f"Found {len(feather_files)} feather file(s) in {coordinates_dir}")

    chunks = []
    for fp in feather_files:
        df = pd.read_feather(fp)
        # Apply each filter; cast both sides to str to handle categorical columns
        mask = pd.Series(True, index=df.index)
        for col, val in filters.items():
            if col not in df.columns:
                print(f"  Warning: column '{col}' missing in {fp.name}, skipping filter")
                continue
            mask &= df[col].astype(str) == str(val)
        matching = df[mask]
        if not matching.empty:
            chunks.append(matching)

    if not chunks:
        raise ValueError(f"No rows matched filters {filters} across all feather files.")

    combined = pd.concat(chunks, ignore_index=True)

    all_flies = list(combined["fly"].dropna().unique())
    print(f"Flies matching filters {filters}: {len(all_flies)}")

    if len(all_flies) < n_flies:
        print(f"Warning: only {len(all_flies)} flies available; using all.")
        sampled_flies = all_flies
    else:
        rng = random.Random(seed)
        sampled_flies = rng.sample(all_flies, n_flies)

    print(f"Sampled {len(sampled_flies)} flies.")
    data = combined[combined["fly"].isin(sampled_flies)].copy()

    data["distance_ball_0_mm"] = data["distance_ball_0"] / PX_PER_MM
    data["distance_ball_0_mm_aligned"] = data.groupby("fly")["distance_ball_0_mm"].transform(
        lambda values: values - values.iloc[0]
    )

    return data


def choose_representative_fly(data: pd.DataFrame, representative_fly: str | None) -> str:
    """Choose representative fly either from user input or by position in unique list."""
    fly_ids = data["fly"].dropna().unique()
    if len(fly_ids) == 0:
        raise ValueError("No fly IDs found in dataset.")

    if representative_fly is not None:
        if representative_fly not in set(fly_ids):
            raise ValueError(
                f"Representative fly '{representative_fly}' not found. " f"Available flies: {sorted(map(str, fly_ids))}"
            )
        return representative_fly

    return default_representative_fly(data)


def default_representative_fly(data: pd.DataFrame) -> str:
    """Return data['fly'].unique()[1] (or [0] if only one fly exists)."""
    fly_ids = data["fly"].dropna().unique()
    return fly_ids[1] if len(fly_ids) > 1 else fly_ids[0]


def get_event_starts(fly_data: pd.DataFrame) -> pd.DataFrame:
    """Get one row per interaction onset event (for event markers)."""
    return (
        fly_data.dropna(subset=["interaction_event_onset"]).groupby("interaction_event_onset", as_index=False).first()
    )


def plot_aligned_trajectories(data: pd.DataFrame, representative_fly_id: str, output_path: Path) -> None:
    """Plot aligned trajectories (gray background), representative in black, onsets in red."""
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 7))

    other_flies = data[data["fly"] != representative_fly_id]
    for _, fly_data in other_flies.groupby("fly"):
        fly_data = fly_data.sort_values("time")
        ax.plot(
            fly_data["time"],
            fly_data["distance_ball_0_mm_aligned"],
            color="gray",
            alpha=0.3,
            linewidth=0.5,
        )

    representative = data[data["fly"] == representative_fly_id].sort_values("time")
    ax.plot(
        representative["time"],
        representative["distance_ball_0_mm_aligned"],
        color="black",
        linewidth=0.8,
    )

    event_starts = get_event_starts(representative)
    if not event_starts.empty:
        ax.scatter(
            event_starts["time"],
            event_starts["distance_ball_0_mm_aligned"],
            color="red",
            marker="x",
            s=50,
            zorder=5,
            label="Contact onset",
        )
        ax.legend(loc="upper left")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ball distance to start (mm)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_interaction_colored_trajectories(data: pd.DataFrame, representative_fly_id: str, output_path: Path) -> None:
    """Plot raw trajectories (gray background) with representative colored by interaction state."""
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(30, 20))

    other_flies = data[data["fly"] != representative_fly_id]
    for _, fly_data in other_flies.groupby("fly"):
        fly_data = fly_data.sort_values("time")
        ax.plot(
            fly_data["time"],
            fly_data["distance_ball_0_mm"],
            color="gray",
            alpha=0.3,
            linewidth=1,
        )

    representative = data[data["fly"] == representative_fly_id].sort_values("time").copy()
    representative["is_interaction"] = representative["interaction_event"].notna()
    representative["segment"] = (
        representative["is_interaction"] != representative["is_interaction"].shift(fill_value=False)
    ).cumsum()

    legend_added = {"red": False, "black": False}
    for _, segment in representative.groupby("segment"):
        color = "red" if segment["is_interaction"].iloc[0] else "black"
        if not legend_added[color]:
            label = "Interaction" if color == "red" else "No interaction"
            legend_added[color] = True
        else:
            label = None
        ax.plot(
            segment["time"],
            segment["distance_ball_0_mm"],
            color=color,
            linewidth=2,
            label=label,
        )

    event_starts = get_event_starts(representative)
    if not event_starts.empty:
        ax.scatter(
            event_starts["time"],
            event_starts["distance_ball_0_mm"],
            color="red",
            marker="x",
            s=100,
            zorder=5,
            label="Contact onset",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ball distance to start (mm)")
    ax.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_colored_trajectories(
    data: pd.DataFrame,
    representative_fly_id: str,
    color_col: str,
    output_path: Path,
) -> None:
    """Aligned trajectories with background flies colored by *color_col*.

    Each unique value in *color_col* gets a distinct color from the tab10
    palette. The representative fly is drawn in black on top.
    Interaction onsets are marked in red.
    """
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_vals = sorted(data[color_col].dropna().astype(str).unique())
    # Use the canonical FeedingState palette; fall back to tab10 for unknown values
    fallback = plt.colormaps["tab10"].resampled(max(len(unique_vals), 1))
    color_map = {val: FEEDING_COLORS.get(val, fallback(i)) for i, val in enumerate(unique_vals)}

    other_flies = data[data["fly"] != representative_fly_id]
    legend_added: set[str] = set()
    for fly_id, fly_data in other_flies.groupby("fly"):
        fly_data = fly_data.sort_values("time")
        val = str(fly_data[color_col].iloc[0]) if color_col in fly_data.columns else "unknown"
        color = color_map.get(val, "gray")
        label = val if val not in legend_added else None
        if label is not None:
            legend_added.add(val)
        ax.plot(
            fly_data["time"],
            fly_data["distance_ball_0_mm_aligned"],
            color=color,
            alpha=0.35,
            linewidth=0.5,
            label=label,
        )

    representative = data[data["fly"] == representative_fly_id].sort_values("time")
    ax.plot(
        representative["time"],
        representative["distance_ball_0_mm_aligned"],
        color="black",
        linewidth=1.2,
        label="Representative",
        zorder=5,
    )

    event_starts = get_event_starts(representative)
    if not event_starts.empty:
        ax.scatter(
            event_starts["time"],
            event_starts["distance_ball_0_mm_aligned"],
            color="red",
            marker="x",
            s=50,
            zorder=6,
            label="Contact onset",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ball distance to start (mm)")
    ax.legend(loc="upper left", fontsize=8, title=color_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_with_histogram(
    data: pd.DataFrame,
    representative_fly_id: str,
    output_path: Path,
) -> None:
    """Publication-ready figure: trajectory panel (52.5×36.6 mm) + right histogram.

    Layout
    ------
    Left  – gray individual fly trajectories (time in min), representative in
             black, contact onsets marked in #f15a29.
    Right – vertical histogram of per-fly final aligned positions (gray bars),
             y-axis shared with the trajectory panel.

    Typography: axis labels 7 pt, tick labels 6 pt.
    """
    # ---- Geometry --------------------------------------------------------
    MAIN_W_MM, MAIN_H_MM = 52.5, 36.6
    HIST_W_MM = 15.0  # width of the histogram side panel
    TOTAL_W = (MAIN_W_MM + HIST_W_MM) / 25.4  # inches
    TOTAL_H = MAIN_H_MM / 25.4

    # Contact marker: 0.5 mm diameter → area in pt²
    # 1 pt = 25.4/72 mm  →  0.5 mm = 0.5*72/25.4 pt
    MARKER_SIZE_PT2 = (0.5 * 72 / 25.4) ** 2
    CONTACT_COLOR = "#f15a29"

    fig, (ax_traj, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(TOTAL_W, TOTAL_H),
        gridspec_kw={"width_ratios": [MAIN_W_MM, HIST_W_MM], "wspace": 0.25},
        sharey=False,
    )

    # ---- Per-fly final aligned position (y at last time point) -----------
    sorted_data = data.sort_values("time")
    final_positions = sorted_data.groupby("fly")["distance_ball_0_mm_aligned"].last().dropna()

    # ---- Trajectory panel ------------------------------------------------
    other_flies = data[data["fly"] != representative_fly_id]
    for _, fly_data in other_flies.groupby("fly"):
        fly_data = fly_data.sort_values("time")
        ax_traj.plot(
            fly_data["time"] / 60.0,
            fly_data["distance_ball_0_mm_aligned"],
            color="gray",
            alpha=0.3,
            linewidth=0.4,
        )

    representative = data[data["fly"] == representative_fly_id].sort_values("time")
    ax_traj.plot(
        representative["time"] / 60.0,
        representative["distance_ball_0_mm_aligned"],
        color="black",
        linewidth=0.8,
    )

    event_starts = get_event_starts(representative)
    if not event_starts.empty:
        ax_traj.scatter(
            event_starts["time"] / 60.0,
            event_starts["distance_ball_0_mm_aligned"],
            color=CONTACT_COLOR,
            marker="x",
            s=MARKER_SIZE_PT2,
            linewidths=0.5,
            zorder=5,
            label="Contact onset",
        )

    ax_traj.set_xlabel("Time (min)", fontsize=7)
    ax_traj.set_ylabel("Ball position (mm)", fontsize=7)
    ax_traj.tick_params(labelsize=6, width=0.5, length=2)
    for spine in ax_traj.spines.values():
        spine.set_linewidth(0.5)
    ax_traj.spines[["top", "right"]].set_visible(False)

    # ---- Histogram panel (horizontal bars, y shared with trajectory) -----
    y_min = min(final_positions.min(), data["distance_ball_0_mm_aligned"].min())
    y_max = max(final_positions.max(), data["distance_ball_0_mm_aligned"].max())
    y_pad = (y_max - y_min) * 0.04
    y_lim = (y_min - y_pad, y_max + y_pad)

    import numpy as np

    n_bins = 20
    counts, edges = np.histogram(final_positions, bins=n_bins, range=(y_lim[0], y_lim[1]))

    ax_hist.barh(
        (edges[:-1] + edges[1:]) / 2,
        counts,
        height=(edges[1] - edges[0]) * 0.85,
        color="gray",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
    )

    ax_hist.tick_params(labelsize=6, width=0.5, length=2)
    ax_hist.set_xlabel("n", fontsize=7)
    for spine in ax_hist.spines.values():
        spine.set_linewidth(0.5)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.yaxis.set_visible(False)

    # Sync y limits
    shared_ylim = (
        min(y_lim[0], ax_traj.get_ylim()[0]),
        max(y_lim[1], ax_traj.get_ylim()[1]),
    )
    ax_traj.set_ylim(shared_ylim)
    ax_hist.set_ylim(shared_ylim)

    plt.tight_layout(pad=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BallTrajectories_2 plots from coordinates feather file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single mode (default) – one feather file:
  python plot_wildtype_trajectories.py
  python plot_wildtype_trajectories.py path/to/file.feather

  # sample mode – draw N flies from the full coordinates directory:
  python plot_wildtype_trajectories.py --mode sample \\
      --filter FeedingState=starved_noWater --filter Light=on --n-flies 53
        """,
    )
    parser.add_argument(
        "feather_path",
        type=Path,
        nargs="?",
        default=DEFAULT_FEATHER,
        help=("[single mode] Path to the coordinates .feather file " f"(default: {DEFAULT_FEATHER.name})"),
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sample"],
        default="single",
        help="'single': load one feather file. 'sample': pool + filter across directory (default: single)",
    )
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=DEFAULT_COORDINATES_DIR,
        help=f"[sample mode] Directory of coordinates feather files (default: {DEFAULT_COORDINATES_DIR})",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="COL=VAL",
        default=[],
        help="[sample mode] Column filter as Key=Value (repeatable), e.g. --filter FeedingState=starved_noWater",
    )
    parser.add_argument(
        "--n-flies",
        type=int,
        default=53,
        help="[sample mode] Number of flies to sample (default: 53)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="[sample mode] Random seed for fly sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where plots are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--representative-fly",
        type=str,
        default=None,
        help="Fly ID to highlight. If omitted, picks fly with most interaction onsets.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="BallTrajectories_2",
        help="Base filename for output plots (default: BallTrajectories_2)",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="FeedingState",
        help="Column used for the colored-trajectories plot (default: FeedingState)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load data -------------------------------------------------------
    if args.mode == "single":
        print(f"Loading dataset: {args.feather_path}")
        data = load_dataset(args.feather_path)
    else:  # sample
        filters: dict[str, str] = {}
        for item in args.filters:
            if "=" not in item:
                raise ValueError(f"--filter must be in Key=Value format, got: '{item}'")
            k, v = item.split("=", 1)
            filters[k.strip()] = v.strip()
        data = load_sampled_dataset(args.coordinates_dir, filters, args.n_flies, seed=args.seed)

    # ---- Pick representative fly -----------------------------------------
    representative_fly_id = choose_representative_fly(data, args.representative_fly)
    print(f"Using representative fly: {representative_fly_id}")

    # ---- Output paths ----------------------------------------------------
    aligned_base = args.output_dir / args.base_name
    interactions_base = args.output_dir / f"{args.base_name}_Withinteractions"
    colored_base = args.output_dir / f"{args.base_name}_ColoredBy_{args.color_by}"

    # ---- Plot 1: aligned, gray background --------------------------------
    plot_aligned_trajectories(data, representative_fly_id, aligned_base)
    print(f"Saved: {aligned_base.with_suffix('.svg')}")
    print(f"Saved: {aligned_base.with_suffix('.eps')}")
    print(f"Saved: {aligned_base.with_suffix('.pdf')}")

    # ---- Plot 2: interaction-colored representative -----------------------
    plot_interaction_colored_trajectories(data, representative_fly_id, interactions_base)
    print(f"Saved: {interactions_base.with_suffix('.svg')}")
    print(f"Saved: {interactions_base.with_suffix('.pdf')}")

    # ---- Plot 3: trajectories colored by column (if column present) ------
    if args.color_by in data.columns:
        plot_colored_trajectories(data, representative_fly_id, args.color_by, colored_base)
        print(f"Saved: {colored_base.with_suffix('.svg')}")
        print(f"Saved: {colored_base.with_suffix('.pdf')}")
    else:
        print(f"Skipping colored plot: column '{args.color_by}' not found in data.")

    # ---- Plot 4: publication figure – trajectories + final-position histogram
    pub_base = args.output_dir / f"{args.base_name}_WithHistogram"
    plot_trajectories_with_histogram(data, representative_fly_id, pub_base)
    print(f"Saved: {pub_base.with_suffix('.svg')}")
    print(f"Saved: {pub_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
