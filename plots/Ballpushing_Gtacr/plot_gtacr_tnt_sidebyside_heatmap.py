#!/usr/bin/env python3
"""Side-by-side Cohen's d heatmaps: TNT screen vs GtACR TH-test.

For each genotype (MB247, TH), computes Cohen's d effect size vs the
respective within-experiment control:
  - TNT screen:  test genotype vs Empty-Gal4
  - GtACR:       OGS32xMB247/TH (BlueLight=on) vs OGS32xEmptySplit (on)

Produces a single figure with two heatmap panels side by side:
  Left  = TNT silencing
  Right = GtACR silencing (light on)

Rows = genotypes (MB247, TH), columns = metrics grouped thematically
(same METRIC_GROUPS as fig3). Asterisks mark FDR-corrected significance.
NaN cells (metric absent in one dataset) are shown in grey.

The figure is comparable to fig3_screen_heatmap for visual cross-validation
of whether the same neurons show a similar phenotype under both silencing
strategies.

Usage:
    python plot_gtacr_tnt_sidebyside_heatmap.py [--n-permutations N]
                                                 [--output-dir PATH]
                                                 [--test]
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from statsmodels.stats.multitest import multipletests

from ballpushing_utils import dataset, read_feather
from ballpushing_utils.plotting import set_illustrator_style

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

fm._load_fontmanager(try_read_cache=False)
set_illustrator_style()
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

GTACR_PATH = dataset(
    "Ballpushing_Gtacr/Datasets/260520_15_summary_Gtacr_TH_Test_Data/summary/pooled_summary.feather"
)
TNT_PATH = dataset(
    "Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Ballpushing_Gtacr/Plots/GtACR_vs_TNT")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "Plotting" / "Config.py"
_spec = importlib.util.spec_from_file_location("Config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not locate Config module at {_CONFIG_PATH}")
Config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Config)

N_PERMUTATIONS = 10_000

# ── Genotype definitions ───────────────────────────────────────────────────────

# Each entry: display label, TNT nickname, GtACR nickname, brain-region key
GENOTYPE_INFO = [
    {
        "display": "MB247",
        "tnt_nickname": "50742 (MB247-GAL4)",
        "gtacr_nickname": "OGS32xMB247",
        "brain_region": "MB",
    },
    {
        "display": "TH",
        "tnt_nickname": "Ple-Gal4.F a.k.a TH-Gal4",
        "gtacr_nickname": "OGS32xTH",
        "brain_region": "Neuropeptide",
    },
]

# ── Metric groups (same as fig3_screen_heatmap) ────────────────────────────────

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
            "speed_trend",
            "normalized_speed",
            "fraction_not_facing_ball",
            "speed_during_interactions",
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
    "normalized_speed": "Normalized walking speed",
    "first_major_event_time": "First major (>1.2mm) event time (s)",
    "max_event_time": "Max ball displ. time (s)",
    "nb_freeze": "Short pauses (>2s <5px) (#)",
    "flailing": "Movement of front legs during contact",
    "speed_during_interactions": "Fly speed during ball contact (mm/s)",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing (>30°) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "speed_trend": "Slope linear fit to fly speed over time",
}

# Metrics that use legacy velocity names → remapped to speed names by read_feather
METRIC_REMAP = {
    "normalized_velocity": "normalized_speed",
    "velocity_during_interactions": "speed_during_interactions",
    "velocity_trend": "speed_trend",
}
# In GtACR dataset, number_of_pauses is stored as nb_pauses
GTACR_METRIC_ALT = {
    "number_of_pauses": "nb_pauses",
}

CLIP_EFFECTS = 1.5

# ── Tile dimensions (same as fig3) ─────────────────────────────────────────────
TILE_W_CM = 0.6284
TILE_H_CM = 0.4206


# ── Statistical helpers ────────────────────────────────────────────────────────


def permutation_test_1d(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    observed = np.abs(group1.mean() - group2.mean())
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = sum(
        1
        for _ in range(n_permutations)
        if np.abs(
            (perm := combined[rng.permutation(len(combined))])[:n1].mean()
            - perm[n1:].mean()
        )
        >= observed
    )
    return (count + 1) / (n_permutations + 1)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


# ── Data loading ───────────────────────────────────────────────────────────────


def load_tnt() -> pd.DataFrame:
    df = read_feather(TNT_PATH)
    df = Config.cleanup_data(df)
    df.rename(
        columns={
            "major_event": "first_major_event",
            "major_event_time": "first_major_event_time",
        },
        inplace=True,
    )
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def load_gtacr() -> pd.DataFrame:
    df = read_feather(GTACR_PATH)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


# ── Analysis ───────────────────────────────────────────────────────────────────


def analyse_tnt_genotype(
    tnt_df: pd.DataFrame,
    tnt_nickname: str,
    col_order: list[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (cohens_d_dict, pval_fdr_dict) for a TNT genotype vs its control."""
    subset = Config.get_subset_data(tnt_df, col="Nickname", value=tnt_nickname)
    if subset.empty:
        return {}, {}
    control_names = [n for n in subset["Nickname"].unique() if n != tnt_nickname]
    if not control_names:
        return {}, {}
    control_name = control_names[0]

    test_rows = subset[subset["Nickname"] == tnt_nickname]
    ctrl_rows = subset[subset["Nickname"] == control_name]

    metrics_tested, pvals, effects = [], [], {}
    for metric in col_order:
        if metric not in subset.columns:
            continue
        g = test_rows[metric].dropna().to_numpy()
        c = ctrl_rows[metric].dropna().to_numpy()
        if len(g) < 2 or len(c) < 2:
            continue
        pvals.append(permutation_test_1d(g, c, n_permutations, rng))
        metrics_tested.append(metric)
        effects[metric] = cohens_d(g, c)

    if not pvals:
        return {}, {}

    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    pvals_dict = {m: float(p) for m, p in zip(metrics_tested, pvals_fdr)}
    return effects, pvals_dict


def analyse_gtacr_genotype(
    gtacr_df: pd.DataFrame,
    gtacr_nickname: str,
    col_order: list[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (cohens_d_dict, pval_fdr_dict) for a GtACR genotype vs EmptySplit (on)."""
    control = "OGS32xEmptySplit"
    on = gtacr_df[gtacr_df["BlueLight"] == "on"]
    test_rows = on[on["Genotype"] == gtacr_nickname]
    ctrl_rows = on[on["Genotype"] == control]

    if test_rows.empty or ctrl_rows.empty:
        return {}, {}

    metrics_tested, pvals, effects = [], [], {}
    for metric in col_order:
        # GtACR may use nb_pauses for number_of_pauses
        gtacr_metric = GTACR_METRIC_ALT.get(metric, metric)
        if gtacr_metric not in on.columns:
            continue
        g = test_rows[gtacr_metric].dropna().to_numpy()
        c = ctrl_rows[gtacr_metric].dropna().to_numpy()
        if len(g) < 2 or len(c) < 2:
            continue
        pvals.append(permutation_test_1d(g, c, n_permutations, rng))
        metrics_tested.append(metric)
        effects[metric] = cohens_d(g, c)

    if not pvals:
        return {}, {}

    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    pvals_dict = {m: float(p) for m, p in zip(metrics_tested, pvals_fdr)}
    return effects, pvals_dict


def build_col_order(
    tnt_df: pd.DataFrame, gtacr_df: pd.DataFrame
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Build ordered column list from METRIC_GROUPS, keeping only metrics in at least one dataset."""
    tnt_cols = set(tnt_df.columns)
    # For GtACR, nb_pauses stands in for number_of_pauses
    gtacr_cols = set(gtacr_df.columns) | {
        "number_of_pauses"
    }  # allow via GTACR_METRIC_ALT

    col_order: list[str] = []
    valid_groups: list[tuple[str, list[str]]] = []
    for group_label, group_keys in METRIC_GROUPS:
        present = [k for k in group_keys if k in tnt_cols or k in gtacr_cols]
        if present:
            col_order.extend(present)
            valid_groups.append((group_label, present))
    return col_order, valid_groups


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_sidebyside_heatmap(
    genotype_results: list[dict],
    col_order: list[str],
    valid_groups: list[tuple[str, list[str]]],
    output_dir: Path,
) -> None:
    """Plot side-by-side TNT | GtACR heatmaps."""
    try:
        color_dict = dict(Config.color_dict)
    except Exception:
        color_dict = {}
    color_dict["Neuropeptide"] = "#8B4513"

    TILE_W_IN = TILE_W_CM / 2.54
    TILE_H_IN = TILE_H_CM / 2.54
    vmin, vmax = -CLIP_EFFECTS, CLIP_EFFECTS

    n_rows = len(genotype_results)
    n_cols = len(col_order)

    # ── Figure geometry ────────────────────────────────────────────────────────
    LEFT_IN = 0.72  # y-axis labels
    GAP_IN = 0.25  # gap between TNT and GtACR panels
    CBAR_GAP_IN = 0.10
    CBAR_W_IN = 0.08
    RIGHT_IN = CBAR_GAP_IN + CBAR_W_IN
    BOTTOM_IN = 1.40  # space for metric labels
    TOP_IN = 0.30  # space for panel titles

    heatmap_w_in = n_cols * TILE_W_IN
    tiles_h_in = n_rows * TILE_H_IN

    # Two panels (TNT + GtACR) side by side, one shared colorbar on the right
    fig_w_in = LEFT_IN + 2 * heatmap_w_in + GAP_IN + RIGHT_IN
    fig_h_in = BOTTOM_IN + tiles_h_in + TOP_IN

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Axes positions (normalised to figure size)
    ax_y = BOTTOM_IN / fig_h_in
    ax_h = tiles_h_in / fig_h_in
    ax_w = heatmap_w_in / fig_w_in

    ax_tnt_x = LEFT_IN / fig_w_in
    ax_gtacr_x = (LEFT_IN + heatmap_w_in + GAP_IN) / fig_w_in

    ax_tnt = fig.add_axes([ax_tnt_x, ax_y, ax_w, ax_h])
    ax_gtacr = fig.add_axes([ax_gtacr_x, ax_y, ax_w, ax_h])

    display_metric_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in col_order]
    row_labels = [g["display"] for g in genotype_results]
    group_colors = ["black", "#888888"]

    def _draw_heatmap(ax, effects_list, pvals_list, is_right: bool):
        """Draw one heatmap panel. effects_list and pvals_list are dicts per row."""
        # Build matrix
        M = np.full((n_rows, n_cols), np.nan)
        P = np.full((n_rows, n_cols), np.nan)
        for i, (effects, pvals) in enumerate(zip(effects_list, pvals_list)):
            for j, metric in enumerate(col_order):
                if metric in effects:
                    M[i, j] = np.clip(effects[metric], -CLIP_EFFECTS, CLIP_EFFECTS)
                if metric in pvals:
                    P[i, j] = pvals[metric]

        M_df = pd.DataFrame(M, index=row_labels, columns=col_order)

        if HAS_SEABORN:
            sns.heatmap(
                M_df,
                ax=ax,
                cmap=plt.get_cmap("RdBu_r"),
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                linewidths=0.4,
                linecolor="lightgray",
                square=False,
                xticklabels=display_metric_names,
                yticklabels=not is_right,
                annot=False,
            )
        else:
            ax.imshow(
                M, cmap=plt.get_cmap("RdBu_r"), vmin=vmin, vmax=vmax, aspect="auto"
            )
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xticklabels(display_metric_names, rotation=45, ha="right")
            ax.set_yticklabels([] if is_right else row_labels)

        # Grey out NaN cells (metrics absent in this dataset)
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(M[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i),
                            1,
                            1,
                            fill=True,
                            color="#cccccc",
                            zorder=2,
                            linewidth=0,
                        )
                    )

        # Significance asterisks
        for i in range(n_rows):
            for j in range(n_cols):
                p = P[i, j]
                if np.isnan(p):
                    continue
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    continue
                cell_val = M[i, j] if not np.isnan(M[i, j]) else 0.0
                text_color = "white" if abs(cell_val) >= 0.5 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.65,
                    stars,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

        # X-tick labels with group colouring
        ax.set_xticklabels(display_metric_names, rotation=45, ha="right", fontsize=6)
        col_to_gc = {}
        for g_idx, (_, g_keys) in enumerate(valid_groups):
            gc = group_colors[g_idx % 2]
            for k in g_keys:
                if k in col_order:
                    col_to_gc[col_order.index(k)] = gc
        for j, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_color(col_to_gc.get(j, "black"))

        # Y-tick labels coloured by brain region (left panel only)
        if not is_right:
            for i, lbl in enumerate(ax.get_yticklabels()):
                region = genotype_results[i]["brain_region"]
                lbl.set_color(color_dict.get(region, "black"))
                lbl.set_rotation(0)
                lbl.set_fontsize(8)

        ax.tick_params(
            which="major", bottom=True, left=not is_right, length=2, direction="out"
        )

    tnt_effects = [g["tnt_effects"] for g in genotype_results]
    tnt_pvals = [g["tnt_pvals"] for g in genotype_results]
    gtacr_effects = [g["gtacr_effects"] for g in genotype_results]
    gtacr_pvals = [g["gtacr_pvals"] for g in genotype_results]

    _draw_heatmap(ax_tnt, tnt_effects, tnt_pvals, is_right=False)
    _draw_heatmap(ax_gtacr, gtacr_effects, gtacr_pvals, is_right=True)

    # Panel titles
    ax_tnt.set_title("TNT silencing", fontsize=9, fontweight="normal", pad=4)
    ax_gtacr.set_title(
        "GtACR silencing\n(OGS32, light on)", fontsize=9, fontweight="normal", pad=4
    )

    # Colorbar
    fig.canvas.draw()
    pos_tnt = ax_tnt.get_position()
    cbar_x = (LEFT_IN + 2 * heatmap_w_in + GAP_IN + CBAR_GAP_IN) / fig_w_in
    ax_cbar = fig.add_axes([cbar_x, ax_y, CBAR_W_IN / fig_w_in, ax_h])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")),
        cax=ax_cbar,
        orientation="vertical",
    )
    cbar.set_label("Cohen's d effect size", fontsize=8)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "gtacr_tnt_sidebyside_heatmap.pdf"
    png_path = output_dir / "gtacr_tnt_sidebyside_heatmap.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test", action="store_true", help="Use fewer permutations for quick testing"
    )
    args = parser.parse_args()

    n_perm = 100 if args.test else args.n_permutations
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    tnt_df = load_tnt()
    gtacr_df = load_gtacr()
    print(f"  TNT: {tnt_df.shape}, GtACR: {gtacr_df.shape}")

    col_order, valid_groups = build_col_order(tnt_df, gtacr_df)
    print(f"  Metric columns: {len(col_order)}")

    genotype_results: list[dict] = []
    rng = np.random.default_rng(42)

    for info in GENOTYPE_INFO:
        display = info["display"]
        print(f"\n── {display} ──────────────────────────────")

        print(f"  TNT: {info['tnt_nickname']}")
        tnt_effects, tnt_pvals = analyse_tnt_genotype(
            tnt_df, info["tnt_nickname"], col_order, n_perm, rng
        )
        print(
            f"    Metrics with effect: {len(tnt_effects)}, significant: {sum(p < 0.05 for p in tnt_pvals.values())}"
        )

        print(f"  GtACR: {info['gtacr_nickname']}")
        gtacr_effects, gtacr_pvals = analyse_gtacr_genotype(
            gtacr_df, info["gtacr_nickname"], col_order, n_perm, rng
        )
        print(
            f"    Metrics with effect: {len(gtacr_effects)}, significant: {sum(p < 0.05 for p in gtacr_pvals.values())}"
        )

        genotype_results.append(
            {
                "display": display,
                "brain_region": info["brain_region"],
                "tnt_effects": tnt_effects,
                "tnt_pvals": tnt_pvals,
                "gtacr_effects": gtacr_effects,
                "gtacr_pvals": gtacr_pvals,
            }
        )

    print("\nPlotting side-by-side heatmap...")
    plot_sidebyside_heatmap(genotype_results, col_order, valid_groups, output_dir)

    # Save stats CSV
    rows = []
    for g in genotype_results:
        for metric in col_order:
            rows.append(
                {
                    "genotype": g["display"],
                    "metric": metric,
                    "tnt_cohens_d": g["tnt_effects"].get(metric, np.nan),
                    "tnt_p_fdr": g["tnt_pvals"].get(metric, np.nan),
                    "gtacr_cohens_d": g["gtacr_effects"].get(metric, np.nan),
                    "gtacr_p_fdr": g["gtacr_pvals"].get(metric, np.nan),
                }
            )
    stats_path = output_dir / "gtacr_tnt_heatmap_statistics.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Saved stats: {stats_path}")
    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
