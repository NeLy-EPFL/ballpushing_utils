#!/usr/bin/env python3
"""
Figure 2 — MagnetBlock: index of first major push (event #).

Permutation test comparing 'no ball access' (n) vs 'access to immobile ball' (y).

Usage:
    python fig2_magnetblock_first_major_push_index.py [--test]
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_PATH = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"
OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Affordance_Figures/Figure2") / Path(__file__).stem
METRIC = "first_major_event"
N_PERMUTATIONS = 10000

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


AX_W_CM = 1.7459
AX_H_CM = 2.2453
BOX_W_CM = 0.4
CENTER_SPACING_CM = 0.94
COLORS = ["#FAA41A", "#3953A4"]
LABEL_COLORS = ["#cb7c29", "#3953A4"]
X_LABELS = ["No access\nto ball", "Access to\nimmobile ball"]


def main(test_mode=False):
    # --- Load data ---
    df = pd.read_feather(DATASET_PATH)

    # Keep only what we need
    df = df[["Magnet", METRIC]].dropna()

    if test_mode:
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
        print(f"Test mode: {len(df)} rows")

    # Raw values (event index, no conversion needed)
    control_raw = df[df["Magnet"] == "n"][METRIC].values
    test_raw = df[df["Magnet"] == "y"][METRIC].values

    # No display conversion — use raw values directly
    control = control_raw
    test = test_raw

    n1, n2 = len(control_raw), len(test_raw)
    print(f"n (no ball): {n1}, y (immobile ball): {n2}")

    # --- Permutation test (median difference, on raw values) ---
    obs_diff = np.median(test_raw) - np.median(control_raw)
    combined = np.concatenate([control_raw, test_raw])
    np.random.seed(42)
    perm_diffs = np.array(
        [np.median((p := np.random.permutation(combined))[n1:]) - np.median(p[:n1]) for _ in range(N_PERMUTATIONS)]
    )
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

    if p_value < 0.001:
        sig_symbol = "***"
    elif p_value < 0.01:
        sig_symbol = "**"
    elif p_value < 0.05:
        sig_symbol = "*"
    else:
        sig_symbol = "ns"

    print(f"Median diff: {obs_diff:.3f}  |  p = {p_value}  |  {sig_symbol}")

    # --- Plot ---
    data_span = AX_W_CM / CENTER_SPACING_CM
    PAD = (data_span - 1) / 2
    box_width = BOX_W_CM / AX_W_CM * data_span

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.set_xlim(-PAD, 1 + PAD)

    positions = [0, 1]
    data_list = [control, test]

    box_kw = dict(linewidth=0.6, color="black")
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(linewidth=0.6, edgecolor="black"),
        whiskerprops=box_kw,
        capprops=box_kw,
        medianprops=dict(linewidth=0.8, color="black"),
        showfliers=False,
    )
    for box in bp["boxes"]:
        box.set_facecolor("none")
        box.set_edgecolor("black")

    rng_jitter = np.random.default_rng(99)
    for pos, y_data, color in zip(positions, data_list, COLORS):
        x_pos = rng_jitter.normal(pos, 0.06, size=len(y_data))
        ax.scatter(x_pos, y_data, s=3, c=color, alpha=1.0, edgecolors="none", zorder=3, clip_on=False)

    # Y axis
    ax.set_ylim([0, 70])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["0", "", "", "", "", "", "", "70"])
    ax.set_ylabel("First major push\n(event #)", fontsize=6, fontname="Arial", labelpad=2)
    ax.tick_params(axis="y", direction="out", length=2, width=0.6, labelsize=6, pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname("Arial")

    # X axis
    ax.set_xticks(positions)
    ax.set_xticklabels(X_LABELS, fontsize=5, fontname="Arial")
    for tick, color in zip(ax.get_xticklabels(), LABEL_COLORS):
        tick.set_color(color)
    ax.tick_params(axis="x", direction="out", length=2, width=0.6, pad=1)

    # Significance bar — placed above ylim, clip_on=False so it renders outside the axes
    y_lo, y_hi = ax.get_ylim()
    y_ax_range = y_hi - y_lo
    bar_y = y_hi + 0.03 * y_ax_range
    ax.plot([0, 1], [bar_y, bar_y], "k-", linewidth=0.6, clip_on=False)
    ax.text(
        0.5,
        bar_y + 0.01 * y_ax_range,
        sig_symbol,
        ha="center",
        va="bottom",
        fontsize=8,
        fontname="Arial",
        clip_on=False,
    )

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    plt.tight_layout()

    # Resize figure so axes are exactly AX_W_CM × AX_H_CM
    ax_pos = ax.get_position()
    new_fig_w_in = (AX_W_CM / 2.54) / ax_pos.width
    new_fig_h_in = (AX_H_CM / 2.54) / ax_pos.height
    fig.set_size_inches(new_fig_w_in, new_fig_h_in)

    # Must be after tight_layout + resize so the layout engine doesn't override it
    ax.yaxis.set_label_coords(-0.1, 0.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / "first_major_event_index_magnetblock.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_pdf}")

    # --- Save stats ---
    stats = {
        "metric": METRIC,
        "control_group": "n",
        "test_group": "y",
        "control_n": n1,
        "test_n": n2,
        "control_median": np.median(control),
        "test_median": np.median(test),
        "median_diff": np.median(test) - np.median(control),
        "p_value": p_value,
        "significance": sig_symbol,
        "n_permutations": N_PERMUTATIONS,
    }
    stats_df = pd.DataFrame([stats])
    stats_csv = OUTPUT_DIR / "first_major_event_index_magnetblock_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved: {stats_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig. 2 MagnetBlock — index of first major push")
    parser.add_argument("--test", action="store_true", help="Run on a small data sample")
    args = parser.parse_args()
    main(test_mode=args.test)
