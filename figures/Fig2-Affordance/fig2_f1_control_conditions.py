#!/usr/bin/env python3
"""Figure 2 — F1 control conditions.

Compares the two ``Pretraining`` groups (n = no training ball,
y = pretrained) on three control metrics:

* ``ball_proximity_proportion_200px`` — proportion of time within 200 px
  of the ball.
* ``nb_events`` — number of interaction events.
* ``distance_moved`` — total ball distance moved (converted to mm).

One permutation test (mean difference) per metric.

Usage:
    python fig2_f1_control_conditions.py [--test]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import (
    paired_boxplot_with_significance,
    resize_axes_cm,
    set_illustrator_style,
    significance_symbol,
)
from ballpushing_utils.plotting.palette import (
    DEFAULT_TWO_GROUP_COLORS,
    DEFAULT_TWO_GROUP_LABEL_COLORS,
)
from ballpushing_utils.stats import permutation_test

DATASET_PATH = dataset(
    "F1_Tracks/Datasets/260121_16_F1_coordinates_F1_New_Data/summary/pooled_summary.feather"
)
N_PERMUTATIONS = 10_000
PIXELS_PER_MM = 500 / 30

CONDITIONS = ("n", "y")  # Pretraining values: no ball, pretrained
BASE_LABELS = {"n": "No train\nball", "y": "Train\nball"}

# Panel dimensions (cm) — pixel-exact assembly in Illustrator.
AX_W_CM = 1.2018
AX_H_CM = 2.3411
BOX_W_CM = 0.3771
CENTER_SPACING_CM = 0.68

METRICS = [
    {
        "col": "ball_proximity_proportion_200px",
        "label": "Time near ball (fraction)",
        "convert": lambda x: x,
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "yticklabels": ["0", "0.2", "0.4", "0.6", "0.8", "1"],
        "ylim": (0.0, 1.05),
    },
    {
        "col": "nb_events",
        "label": "Interaction events (#)",
        "convert": lambda x: x,
        "yticks": [0, 2, 4, 6, 8, 10],
        "yticklabels": None,
        "ylim": (0.0, 10.8),
    },
    {
        "col": "distance_moved",
        "label": "Distance ball moved (mm)",
        "convert": lambda x: x / PIXELS_PER_MM,
        "yticks": [0, 1, 2, 3, 4, 5, 6, 7],
        "yticklabels": None,
        "ylim": (0.0, 7.8),
    },
]


def _plot_metric(df: pd.DataFrame, metric_cfg: dict, p_value: float, output_dir) -> None:
    col = metric_cfg["col"]
    convert = metric_cfg["convert"]

    groups = {c: convert(df.loc[df["Pretraining"] == c, col].dropna().to_numpy()) for c in CONDITIONS}
    ns = {c: len(v) for c, v in groups.items()}

    data_span = AX_W_CM / CENTER_SPACING_CM
    pad = (data_span - 1) / 2
    box_width = BOX_W_CM / AX_W_CM * data_span

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.set_xlim(-pad, 1 + pad)

    paired_boxplot_with_significance(
        ax,
        [groups[c] for c in CONDITIONS],
        p_value=p_value,
        positions=[0, 1],
        colors=DEFAULT_TWO_GROUP_COLORS,
        box_width=box_width,
    )

    # X axis — coloured labels with per-group sample sizes.
    x_labels = [f"{BASE_LABELS[c]} ({ns[c]})" for c in CONDITIONS]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=5, fontname="Arial")
    for tick, color in zip(ax.get_xticklabels(), DEFAULT_TWO_GROUP_LABEL_COLORS):
        tick.set_color(color)
    ax.tick_params(axis="x", direction="out", length=2, width=0.6)

    # Y axis
    ax.set_ylabel(metric_cfg["label"], fontsize=6, fontname="Arial")
    if metric_cfg.get("yticks") is not None:
        ax.set_yticks(metric_cfg["yticks"])
    if metric_cfg.get("yticklabels") is not None:
        ax.set_yticklabels(metric_cfg["yticklabels"])
    if metric_cfg.get("ylim") is not None:
        ax.set_ylim(metric_cfg["ylim"])
    ax.tick_params(axis="y", direction="out", length=2, width=0.6, labelsize=6)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname("Arial")

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

    safe = col.replace("/", "_")
    out_pdf = output_dir / f"{safe}_pretraining.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")


def main(test_mode: bool = False) -> None:
    set_illustrator_style()

    print("Loading dataset...")
    df = read_feather(DATASET_PATH)

    if "Pretraining" not in df.columns:
        raise ValueError(f"'Pretraining' column not found. Columns: {list(df.columns)}")

    # Filter for test ball
    if "ball_identity" in df.columns:
        test_vals = ["test", "Test", "TEST", "1", "test_ball", "testball"]
        df = df[df["ball_identity"].isin(test_vals)].copy()

    # Filter for the F1 conditions included in the paper panel.
    if "F1_condition" in df.columns:
        selected = ["control", "pretrained", "pretrained_unlocked"]
        df = df[df["F1_condition"].isin(selected)].copy()

    df = df[df["Pretraining"].isin(CONDITIONS)].copy()
    print(f"Dataset: {len(df)} rows after filtering")
    for c in CONDITIONS:
        print(f"  {c}: {(df['Pretraining'] == c).sum()} flies")

    if test_mode:
        df = df.sample(n=min(150, len(df)), random_state=42).reset_index(drop=True)
        print(f"Test mode: {len(df)} rows")

    out_dir = figure_output_dir("Figure2", __file__)

    # --- Permutation tests (mean difference, on the converted units) ---
    print(f"\nRunning permutation tests ({N_PERMUTATIONS:,} permutations)...")
    c1, c2 = CONDITIONS
    comparison = f"{c1}_vs_{c2}"
    pvalues: dict[str, float] = {}
    rows: list[dict] = []
    for m_cfg in METRICS:
        col = m_cfg["col"]
        convert = m_cfg["convert"]
        g1 = convert(df.loc[df["Pretraining"] == c1, col].dropna().to_numpy())
        g2 = convert(df.loc[df["Pretraining"] == c2, col].dropna().to_numpy())
        if g1.size < 2 or g2.size < 2:
            p = 1.0
            diff = float("nan")
        else:
            perm = permutation_test(
                g1, g2, statistic="mean", n_permutations=N_PERMUTATIONS, seed=42,
            )
            p = perm.p_value
            # observed_diff is g2 - g1 in the helper (test - control).
            diff = perm.observed_diff
        pvalues[col] = p
        rows.append({
            "metric": col,
            "comparison": comparison,
            "n1": int(g1.size),
            "n2": int(g2.size),
            "mean1": float(np.mean(g1)) if g1.size else float("nan"),
            "mean2": float(np.mean(g2)) if g2.size else float("nan"),
            "mean_diff": diff,
            "p_raw": p,
            "significance": significance_symbol(p),
            "n_permutations": N_PERMUTATIONS,
        })
        print(f"  {col} | {comparison}: p={p:.4f}  {significance_symbol(p)}")

    # --- Save stats CSV ---
    stats_csv = out_dir / "pretraining_stats.csv"
    pd.DataFrame(rows).to_csv(stats_csv, index=False)
    print(f"\nSaved stats: {stats_csv}")

    # --- Plots ---
    for m_cfg in METRICS:
        _plot_metric(df, m_cfg, pvalues[m_cfg["col"]], out_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig. 2 F1 control conditions")
    parser.add_argument("--test", action="store_true", help="Run on a small data sample")
    args = parser.parse_args()
    main(test_mode=args.test)
