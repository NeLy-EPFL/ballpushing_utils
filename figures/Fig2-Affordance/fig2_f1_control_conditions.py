#!/usr/bin/env python3
"""
Figure 2 — F1 control conditions: ball proximity, number of events, ball distance moved.

Compares two Pretraining groups (n = no training ball, y = with training ball) on:
  - ball_proximity_proportion_200px  (proportion of time within 200 px of ball)
  - nb_events                        (number of interaction events)
  - distance_moved                   (total ball distance moved, converted to mm)

Permutation test (median difference) per metric.

Usage:
    python fig2_f1_control_conditions.py [--test]
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_PATH = (
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260121_16_F1_coordinates_F1_New_Data/summary/pooled_summary.feather"
)

OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Affordance_Figures/Figure2") / Path(__file__).stem
N_PERMUTATIONS = 10000
PIXELS_PER_MM = 500 / 30

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

CONDITIONS = ["n", "y"]
COLORS = {"n": "#FAA41A", "y": "#3953A4"}
LABEL_COLORS = {"n": "#cb7c29", "y": "#3953A4"}
BASE_LABELS = {"n": "No train\nball", "y": "Train\nball"}

METRICS = [
    {
        "col": "ball_proximity_proportion_200px",
        "label": "Time near ball (fraction)",
        "convert": lambda x: x,
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "yticklabels": ["0", "0.2", "0.4", "0.6", "0.8", "1"],
        "ylim": [0, 1.05],
    },
    {
        "col": "nb_events",
        "label": "Interaction events (#)",
        "convert": lambda x: x,
        "yticks": [0, 2, 4, 6, 8, 10],
        "yticklabels": None,
        "ylim": [0, 10.8],
    },
    {
        "col": "distance_moved",
        "label": "Distance ball moved (mm)",
        "convert": lambda x: x / PIXELS_PER_MM,
        "yticks": [0, 1, 2, 3, 4, 5, 6, 7],
        "yticklabels": None,
        "ylim": [0, 7.8],
    },
]


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def permutation_test(group1, group2, n_perm=N_PERMUTATIONS):
    obs = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    perm_diffs = np.array(
        [np.mean((p := np.random.permutation(combined))[:n1]) - np.mean(p[n1:]) for _ in range(n_perm)]
    )
    return float(np.mean(np.abs(perm_diffs) >= np.abs(obs)))


def plot_metric(df, metric_cfg, pairwise_results, output_dir):
    col = metric_cfg["col"]
    label = metric_cfg["label"]
    convert = metric_cfg["convert"]

    groups = {c: convert(df[df["Pretraining"] == c][col].dropna().values) for c in CONDITIONS}
    ns = {c: len(v) for c, v in groups.items()}

    # Target axes dimensions in cm
    AX_W_CM = 1.2018
    AX_H_CM = 2.3411
    BOX_W_CM = 0.3771

    # Positions are [0, 1], so 1 data unit = center-to-center distance.
    # Set data span so that 1 data unit maps to CENTER_SPACING_CM on paper.
    CENTER_SPACING_CM = 0.68
    data_span = AX_W_CM / CENTER_SPACING_CM  # total data range over the axis width
    PAD = (data_span - 1) / 2  # equal padding on each side of [0, 1]

    # Box width in data units: computed directly from desired axes width
    box_width = BOX_W_CM / AX_W_CM * data_span

    # Start with a placeholder figure size; will be resized after tight_layout
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.set_xlim(-PAD, 1 + PAD)

    positions = [0, 1]
    data_list = [groups[c] for c in CONDITIONS]

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
    for cond, pos in zip(CONDITIONS, positions):
        vals = groups[cond]
        x_pos = rng_jitter.normal(pos, 0.06, size=len(vals))
        ax.scatter(x_pos, vals, s=3, c=COLORS[cond], alpha=1.0, edgecolors="none", zorder=3)

    key = f"{CONDITIONS[0]}_vs_{CONDITIONS[1]}"
    result = pairwise_results.get(col, {}).get(key, {})
    p_corr = result.get("p_corrected", result.get("p_raw", 1.0))
    sym = sig_label(p_corr)

    # X-axis: colored labels with sample size
    ax.set_xticks(positions)
    x_labels = [f"{BASE_LABELS[c]} ({ns[c]})" for c in CONDITIONS]
    ax.set_xticklabels(x_labels, fontsize=5, fontname="Arial")
    for tick, cond in zip(ax.get_xticklabels(), CONDITIONS):
        tick.set_color(LABEL_COLORS[cond])

    ax.set_ylabel(label, fontsize=6, fontname="Arial")

    # Y-axis ticks, labels, and limits — applied before computing sig bar position
    yticks = metric_cfg.get("yticks")
    if yticks is not None:
        ax.set_yticks(yticks)
    yticklabels = metric_cfg.get("yticklabels")
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ylim_cfg = metric_cfg.get("ylim")
    if ylim_cfg is not None:
        ax.set_ylim(ylim_cfg)

    ax.tick_params(axis="y", direction="out", length=2, width=0.6, labelsize=6)
    ax.tick_params(axis="x", direction="out", length=2, width=0.6)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname("Arial")

    # Significance bar — placed 5% below the top of the y-axis so it is always visible
    y_lo, y_hi = ax.get_ylim()
    y_ax_range = y_hi - y_lo
    bar_y = y_hi - 0.05 * y_ax_range
    ax.plot([0, 1], [bar_y, bar_y], "k-", linewidth=0.6)
    ax.text(0.5, bar_y + 0.01 * y_ax_range, sym, ha="center", va="bottom", fontsize=8, fontname="Arial")

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    plt.tight_layout()

    # Resize figure so the axes are exactly AX_W_CM × AX_H_CM.
    # tight_layout sets axes position as fractions of the figure; scaling the figure
    # scales the axes proportionally (fractions stay fixed), so:
    #   desired_ax_size = ax_fraction * new_fig_size  →  new_fig_size = desired / fraction
    ax_pos = ax.get_position()
    new_fig_w_in = (AX_W_CM / 2.54) / ax_pos.width
    new_fig_h_in = (AX_H_CM / 2.54) / ax_pos.height
    fig.set_size_inches(new_fig_w_in, new_fig_h_in)

    safe = col.replace("/", "_")
    fig.savefig(output_dir / f"{safe}_pretraining.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {safe}_pretraining.pdf")


def main(test_mode=False):
    print("Loading dataset...")
    df = pd.read_feather(DATASET_PATH)

    if "Pretraining" not in df.columns:
        raise ValueError(f"'Pretraining' column not found. Columns: {list(df.columns)}")

    # Filter for test ball
    if "ball_identity" in df.columns:
        test_vals = ["test", "Test", "TEST", "1", "test_ball", "testball"]
        df = df[df["ball_identity"].isin(test_vals)].copy()

    # Filter for F1 conditions (matching comparison script)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Permutation tests ---
    print(f"\nRunning permutation tests ({N_PERMUTATIONS:,} permutations)...")
    pairwise_results = {m["col"]: {} for m in METRICS}

    c1, c2 = CONDITIONS
    key = f"{c1}_vs_{c2}"
    for m_cfg in METRICS:
        col = m_cfg["col"]
        convert = m_cfg["convert"]
        g1 = convert(df[df["Pretraining"] == c1][col].dropna().values)
        g2 = convert(df[df["Pretraining"] == c2][col].dropna().values)
        np.random.seed(42)
        p = permutation_test(g1, g2) if len(g1) >= 2 and len(g2) >= 2 else 1.0
        pairwise_results[col][key] = {"p_raw": p, "p_corrected": p}
        print(f"  {col} | {key}: p={p:.4f}  {sig_label(p)}")

    # --- Save stats CSV ---
    rows = []
    for m_cfg in METRICS:
        col = m_cfg["col"]
        convert = m_cfg["convert"]
        g1 = convert(df[df["Pretraining"] == c1][col].dropna().values)
        g2 = convert(df[df["Pretraining"] == c2][col].dropna().values)
        res = pairwise_results[col][key]
        rows.append(
            {
                "metric": col,
                "comparison": key,
                "n1": len(g1),
                "n2": len(g2),
                "mean1": np.mean(g1),
                "mean2": np.mean(g2),
                "mean_diff": np.mean(g2) - np.mean(g1),
                "p_raw": res["p_raw"],
                "significance": sig_label(res["p_raw"]),
                "n_permutations": N_PERMUTATIONS,
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_csv = OUTPUT_DIR / "pretraining_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nSaved stats: {stats_csv}")

    # --- Plots ---
    for m_cfg in METRICS:
        plot_metric(df, m_cfg, pairwise_results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig. 2 F1 control conditions")
    parser.add_argument("--test", action="store_true", help="Run on a small data sample")
    args = parser.parse_args()
    main(test_mode=args.test)
