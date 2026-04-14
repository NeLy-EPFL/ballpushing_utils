#!/usr/bin/env python3
"""
Figure 3 - TNT screen: brain-region grouped heatmap for selected genotypes.
Hardcoded subset (in order): LC10bc, LC16-1, LH2446, LH272, GR63a, OR67d, DDC, MB247.
Cohen's d effect size heatmap with thematically grouped metrics.
Figure size: 17.5728 cm × 8.5672 cm.
"""

import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

fm._load_fontmanager(try_read_cache=False)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

warnings.filterwarnings("ignore")

sys.path.append("/home/matthias/ballpushing_utils/src/PCA")
import Config

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_PATH = (
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/"
    "250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
)
METRICS_PATH = "/home/matthias/ballpushing_utils/src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"
REGION_MAP_PATH = "/mnt/upramdya_data/MD/Region_map_250908.csv"
OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Affordance_Figures/Figure3") / Path(__file__).stem

# ── TILE DIMENSIONS ───────────────────────────────────────────────────────────
TILE_W_CM = 0.6284  # width per heatmap tile in cm
TILE_H_CM = 0.4206  # height per heatmap tile in cm

# ── EFFECT SIZE CLIP ──────────────────────────────────────────────────────────
CLIP_EFFECTS = 1.5

# ── HARDCODED FIGURE LAYOUT ───────────────────────────────────────────────────
# Each entry: internal_region (key into Config.color_dict), region_display label,
# and entries list of {display: y-axis label, original: exact Nickname in dataset}.
# Using explicit original nicknames avoids errors from the region-map reverse lookup.
FIGURE_LAYOUT = [
    {
        "internal_region": "Vision",
        "region_display": "Visual projection neurons",
        "entries": [
            {"display": "LC10bc", "original": "LC10-2"},
            {"display": "LC16-1", "original": "LC16-1"},
        ],
    },
    {
        "internal_region": "LH",
        "region_display": "Lateral horn neurons",
        "entries": [
            {"display": "LH2446", "original": "75823 (LH2446)"},
            {"display": "LH272", "original": "86685 (LH272)"},
        ],
    },
    {
        "internal_region": "Olfaction",
        "region_display": "Olfactory sensory neurons",
        "entries": [
            {"display": "GR63a", "original": "9943 (GR63a-GAL4)"},
            {"display": "OR67d", "original": "9998 (OR67d-GAL4)"},
        ],
    },
    {
        "internal_region": "Neuropeptide",
        "region_display": "Dopamine & serotonin",
        "entries": [
            {"display": "DDC", "original": "DDC-gal4"},
        ],
    },
    {
        "internal_region": "MB",
        "region_display": "Mushroom body",
        "entries": [
            {"display": "MB247", "original": "50742 (MB247-GAL4)"},
        ],
    },
]

# ── THEMATIC METRIC GROUPS (columns) ──────────────────────────────────────────
METRIC_GROUPS = [
    (
        "Push efficiency",
        ["pulling_ratio", "distance_ratio", "distance_moved", "pulled"],
    ),
    (
        "Ball contact events",
        [
            "max_event",
            "nb_events",
            "first_major_event",
            "first_major_event_time",
            "max_event_time",
            "max_distance",
            "significant_ratio",
            "interaction_persistence",
        ],
    ),
    (
        "Kinematics",
        [
            "velocity_trend",
            "normalized_velocity",
            "fraction_not_facing_ball",
            "velocity_during_interactions",
            "flailing",
            "head_pushing_ratio",
        ],
    ),
    (
        "Pauses & durations",
        [
            "persistence_at_end",
            "time_chamber_beginning",
            "chamber_ratio",
            "chamber_exit_time",
            "number_of_pauses",
            "nb_freeze",
        ],
    ),
]

METRIC_DISPLAY_NAMES = {
    "pulling_ratio": "Proportion pull vs push",
    "distance_ratio": "Dist. ball moved / corridor length",
    "distance_moved": "Dist. ball moved",
    "pulled": "Signif. (>0.3 mm) pulling events (#)",
    "max_event": "Event max. ball displ. (n)",
    "number_of_pauses": "Long pauses (>5s <5px) (#)",
    "first_major_event": "First major (>1.2mm) event (n)",
    "significant_ratio": "Fraction signif. (>0.3 mm) events",
    "max_distance": "Max ball displacement (mm)",
    "chamber_ratio": "Fraction time in chamber",
    "nb_events": "Events (< 2mm fly-ball dist.) (#)",
    "persistence_at_end": "Fraction time near end of corridor",
    "time_chamber_beginning": "Time in chamber first 25% exp. (s)",
    "normalized_velocity": "Normalized walking velocity",
    "first_major_event_time": "First major (>1.2mm) event time (s)",
    "max_event_time": "Max ball displ. time (s)",
    "nb_freeze": "Short pauses (>2s <5px) (#)",
    "flailing": "Movement of front legs during contact",
    "velocity_during_interactions": "Fly speed during ball contact (mm/s)",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing (>30°) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "velocity_trend": "Slope linear fit to fly velocity over time",
}


# ── STATISTICAL HELPERS ────────────────────────────────────────────────────────


def permutation_test_1d(group1, group2, n_permutations=10000, random_state=None):
    rng = np.random.default_rng(random_state)
    observed = np.abs(group1.mean() - group2.mean())
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        pc = combined[perm]
        if np.abs(pc[:n1].mean() - pc[n1:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# ── DATA LOADING ──────────────────────────────────────────────────────────────


def load_data():
    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)
    dataset = dataset[~dataset["Nickname"].isin(["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "MB247-Gal4"])]
    dataset.rename(
        columns={
            "major_event": "first_major_event",
            "major_event_time": "first_major_event_time",
        },
        inplace=True,
    )
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)
    return dataset


def load_metrics():
    with open(METRICS_PATH) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# ── ANALYSIS ──────────────────────────────────────────────────────────────────


def run_analysis(dataset, metrics_list):
    # Build ordered list of (display_label, original_nickname) from FIGURE_LAYOUT
    all_entries = [(e["display"], e["original"]) for grp in FIGURE_LAYOUT for e in grp["entries"]]

    binary_metrics = {"has_finished", "has_major", "has_significant", "nb_significant_events"}

    # Step 1: filter to metrics present with ≤5% NaN (matching plot_detailed_metric_statistics)
    valid_metrics = [
        m for m in metrics_list if m in dataset.columns and m not in binary_metrics and dataset[m].isna().mean() <= 0.05
    ]
    print(f"  Valid metrics: {len(valid_metrics)}")

    # Step 2: global row-drop — remove any row with NaN in any valid metric
    # This matches plot_detailed_metric_statistics exactly, giving identical sample sizes
    rows_with_missing = dataset[valid_metrics].isnull().any(axis=1)
    dataset_clean = dataset[~rows_with_missing].copy()
    print(f"  Rows after global NaN drop: {len(dataset_clean)} (from {len(dataset)})")

    results = []
    for display, original in all_entries:
        if original not in dataset_clean["Nickname"].values:
            print(f"  ⚠  {original!r} not in dataset")
            continue

        subset = Config.get_subset_data(dataset_clean, col="Nickname", value=original)
        if subset.empty:
            continue

        control_names = [n for n in subset["Nickname"].unique() if n != original]
        if not control_names:
            continue
        control_name = control_names[0]

        pvals, metrics_tested, effect_sizes = [], [], {}
        for metric in valid_metrics:
            g = subset[subset["Nickname"] == original][metric].values.astype(float)
            c = subset[subset["Nickname"] == control_name][metric].values.astype(float)
            if len(g) < 2 or len(c) < 2:
                continue
            pvals.append(permutation_test_1d(g, c, random_state=42))
            metrics_tested.append(metric)
            effect_sizes[metric] = cohens_d(g, c)

        if not pvals:
            continue

        _, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        row = {"simplified": display, "original": original, "control": control_name}
        for i, metric in enumerate(metrics_tested):
            row[f"{metric}_cohens_d"] = effect_sizes[metric]
            row[f"{metric}_pval_corrected"] = float(pvals_corr[i])
            row[f"{metric}_significant"] = bool(pvals_corr[i] < 0.05)
        results.append(row)
        print(f"  ✓ {display} ({original}) vs {control_name}")

    return pd.DataFrame(results), valid_metrics


# ── MATRIX BUILDING ───────────────────────────────────────────────────────────


def build_matrix(results_df, valid_metrics):
    valid_set = set(valid_metrics)
    col_order, valid_groups = [], []
    for group_label, group_keys in METRIC_GROUPS:
        present = [k for k in group_keys if k in valid_set]
        if present:
            col_order.extend(present)
            valid_groups.append((group_label, present))
    extra = [m for m in valid_metrics if m not in set(col_order)]
    if extra:
        col_order.extend(extra)
        valid_groups.append(("Other", extra))

    M = pd.DataFrame(0.0, index=results_df["simplified"].tolist(), columns=col_order)
    for _, row in results_df.iterrows():
        for metric in col_order:
            d = row.get(f"{metric}_cohens_d", np.nan)
            if pd.notna(d):
                M.loc[row["simplified"], metric] = float(np.clip(d, -CLIP_EFFECTS, CLIP_EFFECTS))
    return M, col_order, valid_groups


# ── PLOTTING ──────────────────────────────────────────────────────────────────


def plot_figure(results_df, M, col_order, valid_groups):
    try:
        color_dict = dict(Config.color_dict)
    except Exception:
        color_dict = {}
    color_dict["Neuropeptide"] = "#8B4513"

    ROW_FONTSIZE = 8
    COL_FONTSIZE = 6  # metric x-axis labels
    GROUP_FONTSIZE = 7  # thematic group bracket labels

    TILE_W_IN = TILE_W_CM / 2.54
    TILE_H_IN = TILE_H_CM / 2.54

    vmin, vmax = -CLIP_EFFECTS, CLIP_EFFECTS
    display_metric_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in col_order]
    group_colors = ["black", "#888888"]

    # ── Layout geometry ───────────────────────────────────────────────────────
    entries_per_region = [len([e for e in grp["entries"] if e["display"] in M.index]) for grp in FIGURE_LAYOUT]
    n_cols = len(col_order)
    n_total_rows = sum(entries_per_region)
    n_active = sum(1 for e in entries_per_region if e > 0)

    # Fixed margins (inches)
    LEFT_IN = 0.72  # y-axis row labels
    CBAR_GAP_IN = 0.08  # gap between heatmap right edge and colorbar
    CBAR_W_IN = 0.08  # colorbar width
    RIGHT_IN = CBAR_GAP_IN + CBAR_W_IN
    BOTTOM_IN = 1.40  # space for 6pt rotated metric labels + bracket annotations
    TOP_IN = 0.05
    TITLE_H_IN = 0.18  # per-region title above each heatmap block
    GAP_H_IN = 0.01  # gap between bottom of one region and title of next

    heatmap_w_in = n_cols * TILE_W_IN
    tiles_h_in = n_total_rows * TILE_H_IN
    inter_h_in = n_active * TITLE_H_IN + max(0, n_active - 1) * GAP_H_IN

    fig_w_in = LEFT_IN + heatmap_w_in + RIGHT_IN
    fig_h_in = BOTTOM_IN + tiles_h_in + inter_h_in + TOP_IN

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # ── Place heatmap axes manually (bottom→top, last FIGURE_LAYOUT first) ───
    ax_for_region = {}
    y_cursor_in = BOTTOM_IN  # current y in inches from figure bottom

    for orig_idx in range(len(FIGURE_LAYOUT) - 1, -1, -1):
        n_rows = entries_per_region[orig_idx]
        if n_rows == 0:
            ax_for_region[orig_idx] = None
            continue

        ax = fig.add_axes(
            [
                LEFT_IN / fig_w_in,
                y_cursor_in / fig_h_in,
                heatmap_w_in / fig_w_in,
                n_rows * TILE_H_IN / fig_h_in,
            ]
        )
        ax_for_region[orig_idx] = ax
        y_cursor_in += n_rows * TILE_H_IN
        y_cursor_in += TITLE_H_IN  # title space above this region's heatmap
        if orig_idx > 0:
            y_cursor_in += GAP_H_IN  # gap before next region's title

    # Ordered top→bottom (matching FIGURE_LAYOUT order)
    axes = [ax_for_region[i] for i in range(len(FIGURE_LAYOUT))]
    active_axes = [ax for ax in axes if ax is not None]
    axes_bottom = active_axes[-1]
    active_idxs = [i for i, e in enumerate(entries_per_region) if e > 0]

    # ── Draw heatmaps ─────────────────────────────────────────────────────────
    for region_idx, region_info in enumerate(FIGURE_LAYOUT):
        ax = ax_for_region[region_idx]
        if ax is None:
            continue

        region_genotypes = [e["display"] for e in region_info["entries"] if e["display"] in M.index]
        M_region = M.loc[region_genotypes, col_order]
        n_rows = len(region_genotypes)
        region_color = color_dict.get(region_info["internal_region"], "black")
        is_bottom = ax is axes_bottom

        if HAS_SEABORN:
            sns.heatmap(
                M_region,
                ax=ax,
                cmap=plt.get_cmap("RdBu_r"),
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                linewidths=0.4,
                linecolor="lightgray",
                square=False,
                xticklabels=display_metric_names if is_bottom else False,
                yticklabels=True,
                annot=False,
            )
        else:
            ax.imshow(M_region.values, cmap=plt.get_cmap("RdBu_r"), vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xticklabels(display_metric_names if is_bottom else [], rotation=45, ha="right")
            ax.set_yticklabels(region_genotypes)

        ax.tick_params(which="major", bottom=is_bottom, left=True, length=2, direction="out")

        # Significance asterisks
        for i, genotype in enumerate(region_genotypes):
            genotype_row = results_df[results_df["simplified"] == genotype]
            if genotype_row.empty:
                continue
            for j, metric in enumerate(col_order):
                pval = genotype_row.iloc[0].get(f"{metric}_pval_corrected", np.nan)
                if pd.isna(pval):
                    continue
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    continue
                cell_val = float(M_region.iloc[i, j])
                text_color = "white" if abs(cell_val) >= 0.5 else "black"
                ax.text(
                    j + 0.5, i + 0.65, stars, ha="center", va="center", color=text_color, fontsize=11, fontweight="bold"
                )

        # Y-tick labels: colored by brain region
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(0)
            lbl.set_fontsize(ROW_FONTSIZE)
            lbl.set_color(region_color)

        # Region title just above this axes, left-aligned
        ax.text(
            0,
            1.0,
            region_info["region_display"],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="normal",
            color=region_color,
            ha="left",
            va="bottom",
            clip_on=False,
        )

    # ── X-axis labels on bottom subplot ───────────────────────────────────────
    axes_bottom.set_xticklabels(display_metric_names, rotation=45, ha="right", fontsize=COL_FONTSIZE)
    col_to_color = {}
    for g_idx, (_, g_keys) in enumerate(valid_groups):
        gc = group_colors[g_idx % 2]
        for k in g_keys:
            if k in col_order:
                col_to_color[col_order.index(k)] = gc
    for j, lbl in enumerate(axes_bottom.get_xticklabels()):
        lbl.set_color(col_to_color.get(j, "black"))

    # ── Colorbar: spans from first (Visual projection) to second-to-last (DDC) region ───
    if len(active_idxs) >= 2:
        cbar_top_ax = ax_for_region[active_idxs[0]]
        cbar_bottom_ax = ax_for_region[active_idxs[-2]]
    else:
        cbar_top_ax = cbar_bottom_ax = active_axes[0]

    fig.canvas.draw()  # needed before get_position()
    pos_cbar_top = cbar_top_ax.get_position()
    pos_cbar_bot = cbar_bottom_ax.get_position()
    cbar_y1 = pos_cbar_top.y1 - TILE_H_IN / fig_h_in  # top at second row of top region
    cbar_y0 = pos_cbar_bot.y0
    ax_cbar = fig.add_axes(
        [
            (LEFT_IN + heatmap_w_in + CBAR_GAP_IN) / fig_w_in,
            cbar_y0,
            CBAR_W_IN / fig_w_in,
            cbar_y1 - cbar_y0,
        ]
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")),
        cax=ax_cbar,
        orientation="vertical",
    )
    cbar.set_label("Cohen's d effect size", fontsize=8, fontweight="normal")
    ticks = sorted({-CLIP_EFFECTS, -1.0, -0.5, 0.0, 0.5, 1.0, CLIP_EFFECTS})
    cbar.set_ticks(ticks)
    ticklabels = []
    for t in ticks:
        if np.isclose(t, -CLIP_EFFECTS):
            ticklabels.append(f"≤{-CLIP_EFFECTS:g}")
        elif np.isclose(t, CLIP_EFFECTS):
            ticklabels.append(f"≥{CLIP_EFFECTS:g}")
        else:
            ticklabels.append(f"{t:g}")
    cbar.set_ticklabels(ticklabels, fontsize=7)

    # ── Thematic group bracket annotations aligned to label tips ──────────────
    if valid_groups:
        from scipy.optimize import minimize

        col_idx_map = {m: i for i, m in enumerate(col_order)}

        fig.canvas.draw()
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            renderer = fig.canvas.renderer
        fig_inv = fig.transFigure.inverted()

        # Step 1: raw positions from bbox x0 (bottom tip of 45° rotated label)
        tick_labels = axes_bottom.get_xticklabels()
        label_x0_fig = {}
        y_min_fig = float("inf")
        for j, lbl in enumerate(tick_labels):
            bb = lbl.get_window_extent(renderer=renderer)
            pts = fig_inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            label_x0_fig[j] = pts[0, 0]
            y_min_fig = min(y_min_fig, pts[0, 1])

        y_line_fig = y_min_fig - 0.005
        y_label_fig = y_line_fig - 0.010

        # Collect raw bar intervals per group (figure fractions → cm)
        fig_w_cm = fig_w_in * 2.54
        MIN_GAP_CM = 0.45
        MIN_LEN_CM = 1.65
        MAX_LEN_CM = 3.7
        MIN_GAP_FIG = MIN_GAP_CM / fig_w_cm
        MIN_LEN_FIG = MIN_LEN_CM / fig_w_cm
        MAX_LEN_FIG = MAX_LEN_CM / fig_w_cm

        groups_present = []
        for g_idx, (group_label, group_keys) in enumerate(valid_groups):
            present = [k for k in group_keys if k in col_idx_map]
            if not present:
                continue
            j_start = col_idx_map[present[0]]
            j_end = col_idx_map[present[-1]]
            raw_x0 = label_x0_fig.get(j_start, 0.0)
            raw_x1 = label_x0_fig.get(j_end, 0.0)
            groups_present.append((g_idx, group_label, group_keys, raw_x0, raw_x1))

        n_bars = len(groups_present)

        if n_bars > 0:
            # Step 2: enforce minimum length then optimise centers for min gap
            raw_lengths = [gp[4] - gp[3] for gp in groups_present]
            adj_lengths = [min(max(l, MIN_LEN_FIG), MAX_LEN_FIG) for l in raw_lengths]
            raw_centers = [(gp[3] + gp[4]) / 2 for gp in groups_present]

            half = [l / 2 for l in adj_lengths]

            # Max center displacement: ±0.4 cm from raw
            MAX_SHIFT_FIG = 0.4 / fig_w_cm
            bounds = [(rc - MAX_SHIFT_FIG, rc + MAX_SHIFT_FIG) for rc in raw_centers]

            # Objective: stay close to raw centers
            def objective(centers):
                return sum((c - rc) ** 2 for c, rc in zip(centers, raw_centers))

            # Constraints: gap between bar i right-edge and bar i+1 left-edge >= MIN_GAP_FIG
            constraints = []
            for i in range(n_bars - 1):

                def gap_constraint(centers, i=i):
                    return (centers[i + 1] - half[i + 1]) - (centers[i] + half[i]) - MIN_GAP_FIG

                constraints.append({"type": "ineq", "fun": gap_constraint})

            result = minimize(
                objective,
                x0=raw_centers,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            opt_centers = result.x

            # Step 3: draw bars
            for bar_idx, (g_idx, group_label, group_keys, _, _) in enumerate(groups_present):
                cx = opt_centers[bar_idx]
                hl = half[bar_idx]
                x_start = cx - hl
                x_end = cx + hl
                color = group_colors[g_idx % 2]
                fig.add_artist(
                    Line2D(
                        [x_start, x_end],
                        [y_line_fig, y_line_fig],
                        transform=fig.transFigure,
                        color=color,
                        linewidth=1.5,
                        solid_capstyle="butt",
                        clip_on=False,
                    )
                )
                fig.text(
                    (x_start + x_end) / 2,
                    y_label_fig,
                    group_label,
                    transform=fig.transFigure,
                    color=color,
                    ha="center",
                    va="top",
                    fontsize=GROUP_FONTSIZE,
                    fontweight="normal",
                    clip_on=False,
                )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # png = os.path.join(OUTPUT_DIR, "fig3_screen_heatmap.png")
    pdf = os.path.join(OUTPUT_DIR, "fig3_screen_heatmap.pdf")
    # plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    # print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    dataset = load_data()

    print("Loading metrics...")
    metrics_list = load_metrics()

    print("Running analysis...")
    results_df, valid_metrics = run_analysis(dataset, metrics_list)

    if results_df.empty:
        print("No results — check that simplified nicknames exist in the region map.")
        return

    print("Building matrix...")
    M, col_order, valid_groups = build_matrix(results_df, valid_metrics)

    print("Plotting...")
    plot_figure(results_df, M, col_order, valid_groups)

    stats_path = os.path.join(OUTPUT_DIR, "fig3_screen_stats.csv")
    results_df.to_csv(stats_path, index=False)
    print(f"Saved stats: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
