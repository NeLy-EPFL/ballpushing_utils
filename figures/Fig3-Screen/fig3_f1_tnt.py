#!/usr/bin/env python3
"""
Figure 4 – F1 TNT: distance_moved facet plot.

6 genotypes: EmptySplit (control), LC10-2, DDC, TH, TRH, MB247.
Each panel: Naive (gray open box) vs Trained (black open box),
scatter colored by brain region (alpha 0.48 naive / 0.8 trained).
Single shared y-axis. Two-line colored titles. Stats annotated.
"""

import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import set_illustrator_style
from ballpushing_utils.stats import permutation_test

set_illustrator_style()
warnings.filterwarnings("ignore")
fm._load_fontmanager(try_read_cache=False)

# ── PATHS ──────────────────────────────────────────────────────────────────────
DATA_PATH = dataset("F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather")

# ── GENOTYPE CONFIGURATION ─────────────────────────────────────────────────────
GENOTYPE_ORDER = [
    "TNTxEmptySplit",
    "TNTxLC10-2",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
]

GENOTYPE_COLORS = {
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
}

PANEL_TITLES = {
    "TNTxEmptySplit": "Empty-split\n(no silencing)",
    "TNTxLC10-2": "LC10bc silenced\n(visual PNs)",
    "TNTxDDC": "DDC silenced\n(DA & serotonin)",
    "TNTxTH": "TH silenced\n(dopamine)",
    "TNTxTRH": "TRH silenced\n(serotonin)",
    "TNTxMB247": "MB247 silenced\n(mushroom body)",
}

# ── PRETRAINING ────────────────────────────────────────────────────────────────
PRETRAIN_ORDER = ["n", "y"]
PRETRAIN_LABELS = {"n": "Naive", "y": "Trained"}

# box/scatter styles matching the source script's PRETRAINING_STYLES
BOX_STYLES = {
    "n": {"edgecolor": "#7f7f7f", "linewidth": 1.0, "scatter_alpha": 0.48},
    "y": {"edgecolor": "black", "linewidth": 1.0, "scatter_alpha": 0.80},
}

# ── METRIC ─────────────────────────────────────────────────────────────────────
METRIC = "distance_moved"
Y_LABEL = "Distance test ball moved (mm)"
PIXELS_PER_MM = 500 / 30  # 16.67 px per mm — same as reference script

# ── LAYOUT (mm) ────────────────────────────────────────────────────────────────
N_PANELS = len(GENOTYPE_ORDER)
PANEL_W_MM = 22
PANEL_H_MM = 58
GAP_MM = 2.5  # white space between adjacent panels
TITLE_H_MM = 4  # two-line title above each panel
XLABEL_H_MM = 9  # space for x-tick labels
LEFT_MM = 13
RIGHT_MM = 2
TOP_MM = 2
BOTTOM_MM = 2


def _mm(x):
    return x / 25.4


TOTAL_W_MM = LEFT_MM + N_PANELS * PANEL_W_MM + (N_PANELS - 1) * GAP_MM + RIGHT_MM
TOTAL_H_MM = TOP_MM + TITLE_H_MM + PANEL_H_MM + XLABEL_H_MM + BOTTOM_MM
FIG_W = 15.1512 / 2.54  # exact target width in inches
FIG_H = 5.8456 / 2.54  # exact target height in inches

# ── STYLE ──────────────────────────────────────────────────────────────────────
CONTROL_BG = "#f5f5f5"
BOX_WIDTH = 0.55  # width of each box
JITTER = 0.06
SCATTER_S = 12
FONT_TITLE = 6
FONT_LABEL = 7
FONT_TICK = 6
FONT_STARS = 8
N_PERM = 10000
ALPHA_FDR = 0.05


# ── STATISTICS ─────────────────────────────────────────────────────────────────
# FDR correction is applied within each biological group, not across all genotypes.
# Groups reflect which comparisons are scientifically related.
STAT_GROUPS = [
    ["TNTxEmptySplit", "TNTxLC10-2"],  # control + visual (2)
    ["TNTxDDC", "TNTxTH", "TNTxTRH"],  # monoaminergic (3)
    ["TNTxMB247"],  # mushroom body (1)
]


def run_stats(panel_data):
    """Permutation test naive vs trained per genotype, BH-FDR within each brain-region group.

    A single ``np.random.default_rng(42)`` is threaded through every
    call so the joint permutation sequence matches the previously
    published p-values bit-for-bit. ``p_correction="plus_one"`` applies
    the Laplace ``(count + 1) / (n_perm + 1)`` convention used by the
    screen panels.

    Returns a tuple ``(corr_pvals, raw_pvals)`` where both are dicts
    mapping genotype → p-value.
    """
    rng = np.random.default_rng(42)
    result = {}
    raw_result = {}
    for group in STAT_GROUPS:
        pvals, gts = [], []
        for g in group:
            naive = panel_data[g].get("n", np.array([]))
            trained = panel_data[g].get("y", np.array([]))
            if len(naive) >= 2 and len(trained) >= 2:
                p = permutation_test(
                    naive,
                    trained,
                    statistic="mean",
                    n_permutations=N_PERM,
                    rng=rng,
                    p_correction="plus_one",
                ).p_value
                pvals.append(p)
                raw_result[g] = p
                gts.append(g)
        if not pvals:
            continue
        if len(pvals) > 1:
            _, pvals_corr, _, _ = multipletests(pvals, alpha=ALPHA_FDR, method="fdr_bh")
        else:
            pvals_corr = pvals  # single test: no correction needed
        result.update(zip(gts, pvals_corr))
    return result, raw_result


# ── DATA LOADING ───────────────────────────────────────────────────────────────
def prepare_panel_data(df):
    """Convert raw DataFrame to per-genotype, per-pretraining arrays (px → mm)."""
    panel_data = {}
    all_vals = []
    for g in GENOTYPE_ORDER:
        panel_data[g] = {}
        sub = df[df["Genotype"] == g]
        for p in PRETRAIN_ORDER:
            raw = sub[sub["Pretraining"] == p][METRIC].dropna().to_numpy(dtype=float)
            vals = raw / PIXELS_PER_MM
            panel_data[g][p] = vals
            if len(vals):
                all_vals.append(vals)
    return panel_data, all_vals


def build_stats_df(panel_data, raw_pvals, corr_pvals):
    """Assemble a tidy stats DataFrame for the figure."""
    # Derive FDR group label for each genotype
    fdr_group = {}
    for group in STAT_GROUPS:
        label = " | ".join(group)
        for g in group:
            fdr_group[g] = label

    rows = []
    for g in GENOTYPE_ORDER:
        naive_vals = panel_data[g].get("n", np.array([]))
        trained_vals = panel_data[g].get("y", np.array([]))
        p_raw = raw_pvals.get(g, np.nan)
        p_corr = corr_pvals.get(g, np.nan)
        if pd.notna(p_corr):
            annot = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < ALPHA_FDR else "ns"
        else:
            annot = "n/a"
        rows.append(
            {
                "genotype": g,
                "fdr_group": fdr_group.get(g, ""),
                "metric": METRIC,
                "n_naive": len(naive_vals),
                "n_trained": len(trained_vals),
                "mean_naive_mm": float(np.mean(naive_vals)) if len(naive_vals) > 0 else np.nan,
                "mean_trained_mm": float(np.mean(trained_vals)) if len(trained_vals) > 0 else np.nan,
                "p_value_raw": p_raw,
                "p_value_corrected_fdr_bh": p_corr,
                "significant": pd.notna(p_corr) and p_corr < ALPHA_FDR,
                "annotation": annot,
            }
        )
    return pd.DataFrame(rows)


def load_data():
    df = pd.read_feather(DATA_PATH)
    if "ball_identity" in df.columns:
        df = df[df["ball_identity"] == "test"].copy()
    df = df[df["Genotype"].isin(GENOTYPE_ORDER)].copy()
    print(f"  Rows: {len(df)}")
    return df


# ── PLOTTING ───────────────────────────────────────────────────────────────────
def plot_figure(panel_data, all_vals, pval_by_genotype):
    rng = np.random.default_rng(42)

    # Fixed y limits: 0–8 mm, sharp
    y_min = 0.0
    y_max = 8.0

    # Global max data value → shared bar_y for all panels (just above the highest point)
    global_max = max(float(np.nanmax(v)) for v in all_vals if len(v) > 0)
    BAR_Y = global_max + 0.15  # small gap above highest point

    # Pre-compute shared title_y in figure fraction:
    # map (BAR_Y + 0.03 annotation text bottom) from data → figure coords,
    # then add one FONT_STARS line height + a small margin.
    _bottom_f = (BOTTOM_MM + XLABEL_H_MM) / TOTAL_H_MM
    _h_f = PANEL_H_MM / TOTAL_H_MM
    _annot_bot = _bottom_f + _h_f * (BAR_Y + 0.03) / y_max  # annotation text base in fig fraction
    _stars_h = FONT_STARS / 72.0 / FIG_H  # one line of FONT_STARS in fig fraction
    TITLE_Y = _annot_bot + _stars_h + 0.008  # title bottom sits just above annotation

    # x positions: two boxes per panel centered at 1.0 and 2.0
    pos = np.array([1.0, 2.0])
    xpad = BOX_WIDTH * 1.1

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    for i, g in enumerate(GENOTYPE_ORDER):
        color = GENOTYPE_COLORS.get(g, "black")
        is_control = g == "TNTxEmptySplit"

        left_f = (LEFT_MM + i * (PANEL_W_MM + GAP_MM)) / TOTAL_W_MM
        bottom_f = (BOTTOM_MM + XLABEL_H_MM) / TOTAL_H_MM
        w_f = PANEL_W_MM / TOTAL_W_MM
        h_f = PANEL_H_MM / TOTAL_H_MM
        ax = fig.add_axes([left_f, bottom_f, w_f, h_f])
        ax.set_facecolor("white")

        vals_list = [panel_data[g][p] for p in PRETRAIN_ORDER]

        bp = ax.boxplot(
            vals_list,
            positions=pos,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            medianprops=dict(linewidth=1.5, color="black"),
        )

        for patch, p_key in zip(bp["boxes"], PRETRAIN_ORDER):
            style = BOX_STYLES[p_key]
            patch.set_facecolor("none")
            patch.set_edgecolor(style["edgecolor"])
            patch.set_linewidth(style["linewidth"])

        # Scatter
        for vals, pt, p_key in zip(vals_list, pos, PRETRAIN_ORDER):
            if len(vals) == 0:
                continue
            style = BOX_STYLES[p_key]
            xj = rng.normal(pt, JITTER, size=len(vals))
            ax.scatter(xj, vals, s=SCATTER_S, color=color, edgecolors="none", alpha=style["scatter_alpha"], zorder=3)

        # X labels — Naive in gray, Trained in black
        ax.set_xticks(pos)
        tick_texts = ax.set_xticklabels(
            [PRETRAIN_LABELS[p] for p in PRETRAIN_ORDER],
            fontsize=FONT_TICK,
            rotation=35,
            ha="right",
        )
        for txt, p_key in zip(tick_texts, PRETRAIN_ORDER):
            txt.set_color(BOX_STYLES[p_key]["edgecolor"])
        ax.set_xlim(pos[0] - xpad, pos[-1] + xpad)
        ax.set_ylim(y_min, y_max)

        ax.tick_params(which="major", direction="out", length=2.5, width=0.7, labelsize=FONT_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.7)

        ax.set_yticks(list(range(9)))
        ax.spines["left"].set_bounds(0, 8)  # spine stops sharp at 8
        if i == 0:
            ax.spines["left"].set_linewidth(0.7)
            ax.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Significance bar + stars / ns — shared y level just above global data max
        p_corr = pval_by_genotype.get(g)
        if p_corr is not None:
            annot = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < ALPHA_FDR else "ns"
            mid = float(np.mean(pos))
            ax.plot([pos[0], pos[-1]], [BAR_Y, BAR_Y], color="black", lw=0.8, solid_capstyle="butt", zorder=4)
            ax.text(
                mid,
                BAR_Y + 0.03,
                annot,
                ha="center",
                va="bottom",
                fontsize=FONT_STARS,
                fontweight="normal",
                color="black",
            )

        # Two-line colored title in figure coordinates above each panel
        title_x = (LEFT_MM + i * (PANEL_W_MM + GAP_MM) + PANEL_W_MM / 2) / TOTAL_W_MM
        fig.text(
            title_x,
            TITLE_Y,
            PANEL_TITLES[g],
            ha="center",
            va="bottom",
            fontsize=FONT_TITLE,
            color=color,
            multialignment="center",
        )

    out_dir = figure_output_dir("Figure3", __file__)
    path = out_dir / "fig3_f1_tnt.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()
    print("Preparing data...")
    panel_data, all_vals = prepare_panel_data(df)
    print("Running stats...")
    pval_by_genotype, raw_pvals = run_stats(panel_data)
    print("Plotting...")
    plot_figure(panel_data, all_vals, pval_by_genotype)
    out_dir = figure_output_dir("Figure3", __file__)
    stats_df = build_stats_df(panel_data, raw_pvals, pval_by_genotype)
    stats_path = out_dir / "fig3_f1_tnt_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
