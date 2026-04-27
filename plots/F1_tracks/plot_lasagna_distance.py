#!/usr/bin/env python3
"""
Lasagna plots of cumulative ball distance over time for F1 TNT genotypes.

Each row is a single fly sorted by push midpoint time (ascending).
X-axis = adjusted interaction time, colour = cumulative distance.

Outputs (all in OUTPUT_DIR):
  lasagna_<Genotype>.png         — per-genotype, Naive | Pretrained side-by-side
  lasagna_all_genotypes_grid.png — genotypes as rows, conditions as columns
  lasagna_pretraining_effect.png — conditions as rows, genotypes as columns
                                    (best for seeing the pretraining effect)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

COORDINATES_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
)
SUMMARY_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather"
)
OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_Over_Time/Lasagna")

SELECTED_GENOTYPES = [
    "TNTxEmptySplit",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]
CONTROL_GENOTYPE = "TNTxEmptySplit"

CMAP = "YlOrRd"
N_TIMEPOINTS = 600  # resolution of the common time grid
CONTROL_BG = "#f5f5f5"  # faint grey background for control panels


# ============================================================================
# DATA LOADING / PROCESSING  (mirrors compute_distance_moved_over_time_simple.py)
# ============================================================================

def load_datasets():
    print("Loading datasets...")
    df_coords = pd.read_feather(COORDINATES_PATH)
    df_summary = pd.read_feather(SUMMARY_PATH)
    print(f"  F1 coordinates: {df_coords.shape}")
    print(f"  Summary:        {df_summary.shape}")
    return df_coords, df_summary


def compute_distance_over_time_for_fly(df_fly, fly_name, ball_type="test"):
    """Return cumulative distance trace for one fly, or None if insufficient data.

    Uses the 'adjusted_time' column (time within interaction events) when
    present, otherwise falls back to raw 'time'.  Using adjusted_time removes
    the inter-event gaps so the x-axis reflects actual active pushing time.
    """
    distance_col = (
        "training_ball_euclidean_distance" if ball_type == "training"
        else "test_ball_euclidean_distance"
    )
    if "interaction_event" not in df_fly.columns:
        return None

    df_interact = df_fly[df_fly["interaction_event"].notna()].copy()
    if df_interact.empty:
        return None

    # Prefer adjusted_time (cumulative event time) over raw time so the
    # x-axis represents only active interaction time.
    if "adjusted_time" in df_interact.columns and not df_interact["adjusted_time"].isna().all():
        sort_col = "adjusted_time"
        time_label = "Adjusted time (s)"
    else:
        sort_col = "time"
        time_label = "Time (s)"

    df_interact = df_interact.sort_values(sort_col).reset_index(drop=True)
    times = df_interact[sort_col].values.astype(float)
    distances = df_interact[distance_col].values.astype(float)

    if len(times) < 2:
        return None

    distances = np.nan_to_num(distances, nan=0.0)
    cumulative_dist = np.cumsum(distances)

    return {
        "fly_name": fly_name,
        "times": times,
        "full_cumulative": cumulative_dist,
        "total_distance": float(cumulative_dist[-1]),
        "time_label": time_label,
    }


def process_genotype(df_coords, genotype):
    """Process all flies in a genotype, split by pretraining label."""
    df_gen = df_coords[df_coords["Genotype"] == genotype].copy()
    if df_gen.empty or "Pretraining" not in df_gen.columns:
        return {}

    print(f"  {genotype}: {df_gen['fly'].nunique()} flies")
    results = {"genotype": genotype, "pretraining_data": {}}

    for pretrain_val in sorted(df_gen["Pretraining"].dropna().unique()):
        df_pretrain = df_gen[df_gen["Pretraining"] == pretrain_val].copy()
        label = (
            "Pretrained"
            if str(pretrain_val).lower() in {"y", "yes", "true", "1"}
            else "Naive"
        )
        pretrain_results = {"pretraining": label, "flies": {}}

        for fly_name in df_pretrain["fly"].unique():
            r = compute_distance_over_time_for_fly(
                df_pretrain[df_pretrain["fly"] == fly_name], fly_name
            )
            if r is not None:
                pretrain_results["flies"][fly_name] = r

        n = len(pretrain_results["flies"])
        print(f"      {label}: {n} flies")
        if n > 0:
            results["pretraining_data"][label] = pretrain_results

    return results


# ============================================================================
# LASAGNA CORE
# ============================================================================

def _push_midpoint_time(res):
    """
    Time at which the fly reaches 50 % of its total cumulative distance.
    Flies that push early (fast starters) have a small midpoint time.
    Flies that push late or barely at all have a large midpoint time.
    Used as the primary sort key for lasagna rows.
    """
    c = res["full_cumulative"]
    t = res["times"]
    total = res["total_distance"]
    if len(c) == 0 or total == 0:
        return float("inf")
    half = total * 0.5
    idx = int(np.searchsorted(c, half))
    return float(t[min(idx, len(t) - 1)])


def build_matrix(fly_traces, time_grid):
    """
    Interpolate each fly to the common time_grid and stack into a 2-D matrix.
    Primary sort: time to reach 50 % of cumulative distance (early pushers at
    bottom).  Ties broken by total_distance.  Returns (matrix, fly_names).
    """
    sorted_items = sorted(
        fly_traces.items(),
        key=lambda kv: (_push_midpoint_time(kv[1]), -kv[1]["total_distance"]),
    )
    matrix, names = [], []
    for fly_name, res in sorted_items:
        t, c = res["times"], res["full_cumulative"]
        if len(t) < 2:
            row = np.full(len(time_grid), float(c[-1]) if len(c) else 0.0)
        else:
            row = np.interp(time_grid, t, c, left=0.0, right=float(c[-1]))
        matrix.append(row)
        names.append(fly_name)
    return np.array(matrix), names


def draw_lasagna(fly_traces, time_grid, ax, norm, title="", is_control=False):
    """
    Render one lasagna panel.  Returns the AxesImage (for colourbar) or None.
    """
    if is_control:
        ax.set_facecolor(CONTROL_BG)

    # Black spines / panel border
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if not fly_traces:
        ax.text(
            0.5, 0.5, "no data",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=9, color="grey", style="italic",
        )
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold")
        return None

    matrix, _ = build_matrix(fly_traces, time_grid)
    n = len(matrix)

    # infer x-axis label from the first fly result that has a time_label
    time_xlabel = "Time (s)"
    for res in fly_traces.values():
        if "time_label" in res:
            time_xlabel = res["time_label"]
            break

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=CMAP,
        norm=norm,
        extent=[time_grid[0], time_grid[-1], 0, n],
        origin="lower",
        interpolation="nearest",
    )

    ax.set_xlabel(time_xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    # Integer y-ticks make sense (fly index)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return im


# ============================================================================
# GLOBAL NORMALISATION
# ============================================================================

CLIP_PERCENTILE = 95  # clip colour scale at this percentile of per-fly totals


def global_stats(all_results):
    """
    Return (max_time, clip_dist) where clip_dist is the CLIP_PERCENTILE-th
    percentile of per-fly total distances (avoids a few outliers dominating
    the whole colour scale).
    """
    max_t = 0.0
    all_totals = []
    for gen in all_results.values():
        for cond in gen.get("pretraining_data", {}).values():
            for res in cond.get("flies", {}).values():
                if len(res["times"]):
                    max_t = max(max_t, float(res["times"][-1]))
                all_totals.append(res["total_distance"])
    clip_dist = float(np.percentile(all_totals, CLIP_PERCENTILE)) if all_totals else 1.0
    return max_t, clip_dist


# ============================================================================
# OUTPUT 1 – per-genotype plot
# ============================================================================

def plot_per_genotype(gen_results, time_grid, norm, output_dir):
    """Side-by-side Naive | Pretrained lasagna for one genotype."""
    genotype = gen_results["genotype"]
    pdata = gen_results.get("pretraining_data", {})
    conds = ["Naive", "Pretrained"]
    flies_by_cond = {c: pdata.get(c, {}).get("flies", {}) for c in conds}
    present = [c for c in conds if flies_by_cond[c]]

    if not present:
        return

    is_ctrl = genotype == CONTROL_GENOTYPE
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    last_im = None
    for col, cond in enumerate(present):
        ax = axes[col]
        n_flies = len(flies_by_cond[cond])
        im = draw_lasagna(
            flies_by_cond[cond], time_grid, ax, norm,
            title=f"{cond}  (n={n_flies})",
            is_control=is_ctrl,
        )
        # y-label only on the left panel
        if col == 0:
            ax.set_ylabel("Fly (sorted by push midpoint time)", fontsize=9)
        else:
            ax.set_ylabel("")
        if im is not None:
            last_im = im

    # Shared colorbar
    if last_im is not None:
        fig.subplots_adjust(right=0.87, wspace=0.08)
        cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label("Cumulative distance (px)", fontsize=9)

    fig.suptitle(genotype, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.87, 0.95])
    out = output_dir / f"lasagna_{genotype}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Saved {out.name}")
    plt.close()


# ============================================================================
# OUTPUT 2 – all-genotypes grid  (genotype rows × condition columns)
# ============================================================================

def plot_all_grid(all_results, time_grid, norm, output_dir, genotype_order):
    """Rows = genotypes, columns = [Naive, Pretrained]."""
    conds = ["Naive", "Pretrained"]
    genos = [g for g in genotype_order if g in all_results]
    if not genos:
        return

    n_rows, n_cols = len(genos), len(conds)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    last_im = None
    for row, geno in enumerate(genos):
        is_ctrl = geno == CONTROL_GENOTYPE
        pdata = all_results[geno].get("pretraining_data", {})
        for col, cond in enumerate(conds):
            ax = axes[row][col]
            flies = pdata.get(cond, {}).get("flies", {})

            # column header (top row only)
            col_title = cond if row == 0 else ""
            im = draw_lasagna(flies, time_grid, ax, norm,
                               title=col_title, is_control=is_ctrl)
            if im is not None:
                last_im = im

            # y-label: genotype name on the leftmost column only
            if col == 0:
                n_flies = len(flies)
                ax.set_ylabel(f"{geno}\n(n={n_flies})", fontsize=8, labelpad=4)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    # Shared colorbar
    if last_im is not None:
        fig.subplots_adjust(right=0.88, hspace=0.06, wspace=0.06)
        cbar_ax = fig.add_axes([0.90, 0.08, 0.022, 0.84])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label("Cumulative distance (px)", fontsize=10)

    fig.suptitle(
        "Lasagna plots — all genotypes",
        fontsize=13, fontweight="bold", y=1.005,
    )
    out = output_dir / "lasagna_all_genotypes_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close()


# ============================================================================
# OUTPUT 3 – pretraining-effect comparison
#            rows = [Naive, Pretrained], columns = genotypes
#            This is the clearest view of the pretraining effect.
# ============================================================================

def plot_pretraining_effect(all_results, time_grid, norm, output_dir, genotype_order):
    """
    Top row = Naive, bottom row = Pretrained.
    Each column is one genotype, sorted in the order of genotype_order.
    """
    conds = ["Naive", "Pretrained"]
    genos = [g for g in genotype_order if g in all_results]
    if not genos:
        return

    n_rows, n_cols = len(conds), len(genos)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    last_im = None
    for col, geno in enumerate(genos):
        is_ctrl = geno == CONTROL_GENOTYPE
        pdata = all_results[geno].get("pretraining_data", {})
        for row, cond in enumerate(conds):
            ax = axes[row][col]
            flies = pdata.get(cond, {}).get("flies", {})

            # Title: genotype on top row only
            title = geno if row == 0 else ""
            im = draw_lasagna(flies, time_grid, ax, norm,
                               title=title, is_control=is_ctrl)
            if im is not None:
                last_im = im

            # y-axis label: only leftmost column
            if col == 0:
                n_flies = len(flies)
                ax.set_ylabel(
                    f"{cond}\nFly (n={n_flies}, ↓ early pushers)",
                    fontsize=9, labelpad=4,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    # Condition labels as row annotations on the right edge
    for row, cond in enumerate(conds):
        axes[row][-1].annotate(
            cond,
            xy=(1.01, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", va="center",
            rotation=-90,
        )

    # Shared colorbar
    if last_im is not None:
        fig.subplots_adjust(right=0.88, hspace=0.08, wspace=0.06)
        cbar_ax = fig.add_axes([0.90, 0.08, 0.022, 0.84])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label("Cumulative distance (px)", fontsize=10)

    fig.suptitle(
        "Pretraining effect — Naive (top) vs Pretrained (bottom)",
        fontsize=13, fontweight="bold", y=1.005,
    )
    out = output_dir / "lasagna_pretraining_effect.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("LASAGNA PLOTS — CUMULATIVE DISTANCE OVER TIME")
    print("=" * 70 + "\n")

    df_coords, _ = load_datasets()

    required = ["interaction_event", "test_ball_euclidean_distance"]
    missing = [c for c in required if c not in df_coords.columns]
    if missing:
        print(f"Error: missing columns: {missing}")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nProcessing genotypes...")
    all_results = {}
    for geno in SELECTED_GENOTYPES:
        r = process_genotype(df_coords, geno)
        if r:
            all_results[geno] = r

    if not all_results:
        print("Error: no results generated.")
        return False

    print("\nComputing global normalisation...")
    max_time, clip_dist = global_stats(all_results)
    print(f"  Max time:          {max_time:.1f} s")
    print(f"  Colour scale clip: {clip_dist:.0f} px  ({CLIP_PERCENTILE}th percentile)")

    time_grid = np.linspace(0, max_time, N_TIMEPOINTS)
    # PowerNorm with gamma<1 further compresses the high end after clipping,
    # making gradations among the majority of flies more visible.
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=clip_dist)

    print("\nGenerating per-genotype lasagna plots...")
    for geno in SELECTED_GENOTYPES:
        if geno in all_results:
            plot_per_genotype(all_results[geno], time_grid, norm, OUTPUT_DIR)

    print("\nGenerating all-genotypes grid...")
    plot_all_grid(all_results, time_grid, norm, OUTPUT_DIR, SELECTED_GENOTYPES)

    print("\nGenerating pretraining-effect comparison...")
    plot_pretraining_effect(all_results, time_grid, norm, OUTPUT_DIR, SELECTED_GENOTYPES)

    print("\n" + "=" * 70)
    print("Done!  Results saved to:", OUTPUT_DIR)
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
