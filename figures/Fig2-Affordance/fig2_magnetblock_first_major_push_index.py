#!/usr/bin/env python3
"""Figure 2 — MagnetBlock: index of first major push (event #).

Permutation test comparing 'no ball access' (Magnet=='n') vs
'access to immobile ball' (Magnet=='y').

Usage:
    python fig2_magnetblock_first_major_push_index.py [--test]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import (
    paired_boxplot_with_significance,
    resize_axes_cm,
    set_illustrator_style,
)
from ballpushing_utils.plotting.palette import DEFAULT_TWO_GROUP_LABEL_COLORS
from ballpushing_utils.stats import permutation_test

DATASET_PATH = dataset(
    "MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"
)
METRIC = "first_major_event"
N_PERMUTATIONS = 10_000

# Panel dimensions (cm) — match the time-variant so panels align in figure.
AX_W_CM = 1.7459
AX_H_CM = 2.2453
BOX_W_CM = 0.4
CENTER_SPACING_CM = 0.94

X_LABELS = ("No access\nto ball", "Access to\nimmobile ball")


def main(test_mode: bool = False) -> None:
    set_illustrator_style()

    # --- Load ---
    df = pd.read_feather(DATASET_PATH)[["Magnet", METRIC]].dropna()
    if test_mode:
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
        print(f"Test mode: {len(df)} rows")

    control_raw = df.loc[df["Magnet"] == "n", METRIC].to_numpy()
    test_raw = df.loc[df["Magnet"] == "y", METRIC].to_numpy()
    n1, n2 = control_raw.size, test_raw.size
    print(f"n (no ball): {n1}, y (immobile ball): {n2}")

    # --- Permutation test (median diff on raw event indices) ---
    perm = permutation_test(
        control_raw,
        test_raw,
        statistic="median",
        n_permutations=N_PERMUTATIONS,
        seed=42,
    )
    print(f"Median diff: {perm.observed_diff:.3f}  |  p = {perm.p_value}")

    # Event indices plot directly on the raw scale.
    control = control_raw
    test = test_raw

    # --- Plot ---
    data_span = AX_W_CM / CENTER_SPACING_CM
    pad = (data_span - 1) / 2
    box_width = BOX_W_CM / AX_W_CM * data_span

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.set_xlim(-pad, 1 + pad)

    paired_boxplot_with_significance(
        ax,
        [control, test],
        p_value=perm.p_value,
        positions=[0, 1],
        box_width=box_width,
    )

    # Y axis
    ax.set_ylim([0, 70])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["0", "", "", "", "", "", "", "70"])
    ax.set_ylabel("First major push\n(event #)", fontsize=6, fontname="Arial", labelpad=2)
    ax.tick_params(axis="y", direction="out", length=2, width=0.6, labelsize=6, pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname("Arial")

    # X axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(X_LABELS, fontsize=5, fontname="Arial")
    for tick, color in zip(ax.get_xticklabels(), DEFAULT_TWO_GROUP_LABEL_COLORS):
        tick.set_color(color)
    ax.tick_params(axis="x", direction="out", length=2, width=0.6, pad=1)

    # Frame
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    plt.tight_layout()
    resize_axes_cm(fig, ax, AX_W_CM, AX_H_CM)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # --- Save figure + stats ---
    out_dir = figure_output_dir("Figure2", __file__)
    out_pdf = out_dir / "first_major_event_index_magnetblock.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")

    stats = {
        "metric": METRIC,
        "control_group": "n",
        "test_group": "y",
        "control_n": n1,
        "test_n": n2,
        "control_median": float(np.median(control)),
        "test_median": float(np.median(test)),
        "median_diff": float(np.median(test) - np.median(control)),
        "p_value": perm.p_value,
        "n_permutations": perm.n_permutations,
    }
    stats_csv = out_dir / "first_major_event_index_magnetblock_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)
    print(f"Saved: {stats_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig. 2 MagnetBlock — index of first major push")
    parser.add_argument("--test", action="store_true", help="Run on a small data sample")
    args = parser.parse_args()
    main(test_mode=args.test)
