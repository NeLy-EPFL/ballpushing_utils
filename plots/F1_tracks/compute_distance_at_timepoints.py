#!/usr/bin/env python3
"""
Compute distance moved at specific time bins for F1 TNT genotypes.

Reads distance_moved_Xmin columns directly from the summary dataset
(computed by BallPushingMetrics.get_distance_moved_at_timepoint during
dataset building). Creates boxplots grouped by pretraining and runs
permutation tests.

Usage:
    python compute_distance_at_timepoints.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SUMMARY_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260325_16_summary_F1_TNT_Full_Data/summary/pooled_summary.feather"
)

OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_At_Timepoints")

SELECTED_GENOTYPES = [
    "TNTxEmptySplit",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

# Column name → display label for each 10-min interval of the test phase
TIMEPOINT_COLS = [
    "distance_moved_10min",
    "distance_moved_20min",
    "distance_moved_30min",
    "distance_moved_40min",
    "distance_moved_50min",
    "distance_moved_60min",
]
TIME_BIN_LABELS = ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"]
COL_TO_LABEL = dict(zip(TIMEPOINT_COLS, TIME_BIN_LABELS))
LABEL_TO_COL = dict(zip(TIME_BIN_LABELS, TIMEPOINT_COLS))

# Equivalent raw-seconds cutoffs (used only for labelling / CSV export)
TIME_BIN_SECS = [4200, 4800, 5400, 6000, 6600, 7200]

# Color scheme matching f1_tnt_genotype_comparison.py
GENOTYPE_COLORS = {
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
}

# Pretraining styles matching f1_tnt_genotype_comparison.py
PRETRAINING_STYLES = {
    "Naive": {"alpha": 0.6, "edgecolor": "black", "linewidth": 1.5},
    "Pretrained": {"alpha": 1.0, "edgecolor": "black", "linewidth": 2.5},
}

# Pixel to mm conversion
PIXELS_PER_MM = 500 / 30

# Plot styling
JITTER_AMOUNT = 0.15
SCATTER_SIZE = 40
SCATTER_ALPHA = 0.5

# Permutation test settings
N_PERMUTATIONS = 5000

# ============================================================================
# DATA LOADING
# ============================================================================


def load_and_prepare():
    """
    Load the summary feather, filter to test-ball rows for selected genotypes,
    and return a long-format DataFrame with one row per fly × timepoint.

    Distances are stored in pixels in the summary; this function converts them
    to mm for all downstream use.
    """
    print("Loading summary dataset...")
    df = pd.read_feather(SUMMARY_PATH)
    print(f"  Full summary: {df.shape}")

    # Keep only test-ball rows
    df = df[df["ball_identity"] == "test"].copy()
    print(f"  Test-ball rows: {len(df)}")

    # Keep selected genotypes
    df = df[df["Genotype"].isin(SELECTED_GENOTYPES)].copy()
    print(f"  After genotype filter: {len(df)} rows, {df['fly'].nunique()} flies")

    # Normalise Pretraining labels
    df["pretraining"] = df["Pretraining"].map(
        lambda v: "Pretrained" if str(v).strip().lower() in ("y", "yes", "true", "1") else "Naive"
    )

    # Check all timepoint columns are present
    missing = [c for c in TIMEPOINT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing timepoint columns: {missing}")

    # Rename to lowercase for consistent downstream use
    df = df.rename(columns={"Genotype": "genotype"})

    # Melt to long format (pixels)
    df_long = df.melt(
        id_vars=["fly", "genotype", "pretraining", "distance_moved"],
        value_vars=TIMEPOINT_COLS,
        var_name="timepoint_col",
        value_name="distance_pixels",
    )
    df_long["timepoint"] = df_long["timepoint_col"].map(COL_TO_LABEL)
    df_long["timepoint_sec"] = df_long["timepoint_col"].map(
        dict(zip(TIMEPOINT_COLS, TIME_BIN_SECS))
    )
    df_long["distance_mm"] = df_long["distance_pixels"] / PIXELS_PER_MM
    df_long["distance_moved_mm"] = df_long["distance_moved"] / PIXELS_PER_MM

    print(f"  Long-format records: {len(df_long)}")
    return df_long


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def _sig_label(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def perform_permutation_tests_at_timepoints(df_results, n_permutations=N_PERMUTATIONS):
    """
    For each genotype × timepoint, test Pretrained vs Naive distance with a
    2-sample permutation test on the means.

    Multiple-comparison correction uses Holm-Bonferroni (method="holm"), which
    controls the familywise error rate under arbitrary dependence between tests.
    This is appropriate here because the six timepoints are tested on the *same*
    animals and the cumulative values at later timepoints contain the earlier
    ones, making the tests positively correlated.

    Returns
    -------
    dict
        {genotype: {timepoint_sec: {"p": float, "p_fdr": float, "sig": str,
         "n_naive": int, "n_pretrained": int, "delta_mm": float}}}
    """
    all_stats = {}

    for genotype in sorted(df_results["genotype"].unique()):
        df_gen = df_results[df_results["genotype"] == genotype]
        timepoints = sorted(df_gen["timepoint_sec"].unique())

        raw_p = {}
        delta = {}
        n_naive_map = {}
        n_pretrained_map = {}

        rng = np.random.default_rng(42)

        for tp in timepoints:
            df_tp = df_gen[df_gen["timepoint_sec"] == tp]
            naive = df_tp[df_tp["pretraining"] == "Naive"]["distance_mm"].dropna().values
            pretrained = df_tp[df_tp["pretraining"] == "Pretrained"]["distance_mm"].dropna().values

            n_naive_map[tp] = len(naive)
            n_pretrained_map[tp] = len(pretrained)

            if len(naive) < 2 or len(pretrained) < 2:
                raw_p[tp] = 1.0
                delta[tp] = np.nan
                continue

            obs_diff = float(np.mean(pretrained) - np.mean(naive))
            delta[tp] = obs_diff

            combined = np.concatenate([naive, pretrained])
            n1 = len(naive)
            # Vectorised: permute all rows at once (~50-100× faster than loop)
            perm_idx = rng.permuted(
                np.tile(np.arange(len(combined)), (n_permutations, 1)), axis=1
            )
            perm_diffs = (
                combined[perm_idx[:, n1:]].mean(axis=1)
                - combined[perm_idx[:, :n1]].mean(axis=1)
            )

            p = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (n_permutations + 1)
            raw_p[tp] = float(p)

        # Holm-Bonferroni correction within this genotype across timepoints
        tps = list(raw_p.keys())
        p_vals = [raw_p[tp] for tp in tps]
        if len(p_vals) > 1:
            _, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
        else:
            p_corr = p_vals

        geno_stats = {}
        for tp, p, pf in zip(tps, p_vals, p_corr):
            geno_stats[tp] = {
                "p": round(p, 6),
                "p_fdr": round(float(pf), 6),
                "sig": _sig_label(float(pf)),
                "n_naive": n_naive_map[tp],
                "n_pretrained": n_pretrained_map[tp],
                "delta_mm": round(delta.get(tp, float("nan")), 3),
            }

        all_stats[genotype] = geno_stats
        print(f"  {genotype}: " + "  ".join(
            f"{TIME_BIN_LABELS[TIME_BIN_SECS.index(tp)]} p={geno_stats[tp]['p_fdr']:.4f} {geno_stats[tp]['sig']}"
            for tp in tps if tp in TIME_BIN_SECS
        ))

    return all_stats


def export_permutation_stats(all_stats, output_dir):
    """Save permutation test results to CSV."""
    tp_label_map = dict(zip(TIME_BIN_SECS, TIME_BIN_LABELS))
    records = []
    for genotype, geno_stats in all_stats.items():
        for tp_sec, s in geno_stats.items():
            records.append({
                "genotype": genotype,
                "timepoint_sec": int(tp_sec),
                "timepoint_label": tp_label_map.get(tp_sec, str(tp_sec)),
                "n_naive": s["n_naive"],
                "n_pretrained": s["n_pretrained"],
                "delta_mm": s["delta_mm"],
                "p_value": s["p"],
                "p_fdr": s["p_fdr"],
                "significance": s["sig"],
            })
    df_out = pd.DataFrame(records)
    out_path = Path(output_dir) / "permutation_test_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved permutation results: {out_path.name}")
    return df_out


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def _draw_significance_bracket(ax, x1, x2, y, sig, fontsize=10):
    """Draw a bracket with significance label between x1 and x2 at height y."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + h * 1.1, sig,
            ha="center", va="bottom", fontsize=fontsize,
            color="black" if sig != "ns" else "#888888")


def compute_global_ylim(df_results, data_percentile=99, bracket_headroom=0.22):
    """
    Compute a single (y_min, y_max) that will be applied to every plot so that
    all axes share the same scale — making grid comparisons meaningful.

    Parameters
    ----------
    data_percentile : int
        Upper percentile of *distance_mm* used as the data ceiling.
    bracket_headroom : float
        Fraction of the data ceiling reserved above the data for significance
        brackets.  0.22 gives ~22% headroom.

    Returns
    -------
    (float, float)  —  (y_min, y_top)
    """
    vals = df_results["distance_mm"].dropna().values
    if len(vals) == 0:
        return (0.0, 1.0)
    data_ceil = float(np.percentile(vals, data_percentile))
    y_top = data_ceil * (1 + bracket_headroom)
    return (0.0, y_top)


def _apply_ylim(ax, ylim):
    """Set y-axis limits, keeping a small gap below 0."""
    ax.set_ylim(ylim[0] - ylim[1] * 0.02, ylim[1])


def plot_distance_by_timepoint(df_results, output_dir, stats=None, ylim=None):
    """
    Create boxplots with jittered scatter for each timepoint, all genotypes combined.
    Optionally overlays significance brackets (stats dict from
    perform_permutation_tests_at_timepoints).
    Returns paths to saved plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}
    n_genotypes = df_results["genotype"].nunique()
    jitter = JITTER_AMOUNT * (3 / max(n_genotypes, 3))  # Adjust for number of genotypes

    # Derive bracket anchor from ylim so brackets sit at a fixed fractional height
    if ylim is not None:
        _y_data_ceil = ylim[1] / 1.22  # reverse the 22% headroom
    else:
        all_vals = df_results["distance_mm"].dropna().values
        _y_data_ceil = float(np.percentile(all_vals, 99)) if len(all_vals) else 1.0

    for timepoint, time_label in zip(TIME_BIN_SECS, TIME_BIN_LABELS):
        df_time = df_results[df_results["timepoint"] == time_label].copy()

        if df_time.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        genotypes = sorted(df_time["genotype"].unique())
        pretraining_vals = sorted(df_time["pretraining"].unique())

        # Create positions for grouped boxplots
        n_groups = len(genotypes)
        n_conditions = len(pretraining_vals)
        group_width = 0.8
        box_width = group_width / n_conditions

        x_positions = np.arange(n_groups)
        box_centres = {}

        # Plot for each pretraining condition
        for i, pretrain in enumerate(pretraining_vals):
            df_pretrain = df_time[df_time["pretraining"] == pretrain]

            box_positions = x_positions + (i - n_conditions / 2 + 0.5) * box_width
            for j_pos, g in enumerate(genotypes):
                box_centres[(g, pretrain)] = box_positions[j_pos]

            # Prepare data for boxplot
            data_by_genotype = [df_pretrain[df_pretrain["genotype"] == g]["distance_mm"].values for g in genotypes]

            # Get color (use first genotype's color for now, will customize per box)
            pretrain_style = PRETRAINING_STYLES.get(pretrain, {})

            # Create boxplots
            bp = ax.boxplot(
                data_by_genotype,
                positions=box_positions,
                widths=box_width * 0.6,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(alpha=pretrain_style.get("alpha", 0.8)),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Color boxes by genotype
            for patch, genotype in zip(bp["boxes"], genotypes):
                color = GENOTYPE_COLORS.get(genotype, "#808080")
                patch.set_facecolor(color)
                patch.set_edgecolor(pretrain_style.get("edgecolor", "black"))
                patch.set_linewidth(pretrain_style.get("linewidth", 1.5))

            # Add jittered scatter points
            for j, genotype in enumerate(genotypes):
                data = df_pretrain[df_pretrain["genotype"] == genotype]["distance_mm"].values

                if len(data) == 0:
                    continue

                # Add jitter
                x_jitter = np.random.normal(box_positions[j], jitter, len(data))

                color = GENOTYPE_COLORS.get(genotype, "#808080")
                ax.scatter(
                    x_jitter,
                    data,
                    s=SCATTER_SIZE,
                    alpha=SCATTER_ALPHA,
                    color=color,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )

        # Significance annotations
        if stats and "Naive" in pretraining_vals and "Pretrained" in pretraining_vals:
            bracket_y_base = _y_data_ceil * 1.03
            for genotype_sig in genotypes:
                if genotype_sig not in stats or timepoint not in stats[genotype_sig]:
                    continue
                sig = stats[genotype_sig][timepoint]["sig"]
                x_naive = box_centres.get((genotype_sig, "Naive"))
                x_pre = box_centres.get((genotype_sig, "Pretrained"))
                if x_naive is None or x_pre is None:
                    continue
                _draw_significance_bracket(ax, x_naive, x_pre, bracket_y_base, sig, fontsize=9)

        # Apply shared y-axis limits
        if ylim is not None:
            _apply_ylim(ax, ylim)

        # Customize plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(genotypes, rotation=45, ha="right")
        ax.set_ylabel("Distance Moved (mm)", fontsize=13, fontweight="bold")
        ax.set_title(f"Distance Moved at {time_label} ({timepoint} seconds)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Create legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                edgecolor=PRETRAINING_STYLES[p].get("edgecolor", "black"),
                linewidth=PRETRAINING_STYLES[p].get("linewidth", 1.5),
                alpha=PRETRAINING_STYLES[p].get("alpha", 0.8),
                label=p,
            )
            for p in pretraining_vals
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

        plt.tight_layout()

        # Save
        output_path = output_dir / f"distance_at_{timepoint}sec.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plot_paths[time_label] = output_path
        print(f"  Saved: {output_path.name}")
        plt.close()

    return plot_paths


def plot_distance_by_genotype(df_results, output_dir, stats=None, ylim=None):
    """
    Create individual plots for each genotype showing all timepoints.
    Optionally overlays permutation-test significance brackets (stats dict from
    perform_permutation_tests_at_timepoints).
    Returns paths to saved plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    for genotype in sorted(df_results["genotype"].unique()):
        df_gen = df_results[df_results["genotype"] == genotype].copy()

        if df_gen.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        timepoints = sorted(df_gen["timepoint_sec"].unique())
        # Always put Naive before Pretrained for consistent ordering
        all_pretrain = sorted(df_gen["pretraining"].unique())
        pretraining_vals = ([p for p in ["Naive", "Pretrained"] if p in all_pretrain]
                            + [p for p in all_pretrain if p not in ["Naive", "Pretrained"]])

        # Create positions
        n_timepoints = len(timepoints)
        n_conditions = len(pretraining_vals)
        group_width = 0.8
        box_width = group_width / n_conditions

        x_positions = np.arange(n_timepoints)

        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")

        # Track box centre positions keyed by (timepoint_sec, pretraining_label)
        box_centres = {}

        # Plot for each pretraining condition
        for i, pretrain in enumerate(pretraining_vals):
            df_pretrain = df_gen[df_gen["pretraining"] == pretrain]

            box_positions = x_positions + (i - n_conditions / 2 + 0.5) * box_width

            for j, tp in enumerate(timepoints):
                box_centres[(tp, pretrain)] = box_positions[j]

            # Prepare data
            data_by_time = [df_pretrain[df_pretrain["timepoint_sec"] == t]["distance_mm"].values for t in timepoints]

            pretrain_style = PRETRAINING_STYLES.get(pretrain, {})

            # Create boxplots
            ax.boxplot(
                data_by_time,
                positions=box_positions,
                widths=box_width * 0.6,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=genotype_color,
                    alpha=pretrain_style.get("alpha", 0.8),
                    edgecolor=pretrain_style.get("edgecolor", "black"),
                    linewidth=pretrain_style.get("linewidth", 1.5),
                ),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Add jittered scatter
            for j, timepoint_sec in enumerate(timepoints):
                data = df_pretrain[df_pretrain["timepoint_sec"] == timepoint_sec]["distance_mm"].values

                if len(data) == 0:
                    continue

                x_jitter = np.random.normal(box_positions[j], JITTER_AMOUNT * 0.5, len(data))

                ax.scatter(
                    x_jitter,
                    data,
                    s=SCATTER_SIZE,
                    alpha=SCATTER_ALPHA,
                    color=genotype_color,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )

        # Customize
        ax.set_xticks(x_positions)
        ax.set_xticklabels(TIME_BIN_LABELS, rotation=0)
        ax.set_xlabel("Time Point", fontsize=13, fontweight="bold")
        ax.set_ylabel("Distance Moved (mm)", fontsize=13, fontweight="bold")
        ax.set_title(f"{genotype} - Distance Moved Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=genotype_color,
                edgecolor=PRETRAINING_STYLES.get(p, {}).get("edgecolor", "black"),
                linewidth=PRETRAINING_STYLES.get(p, {}).get("linewidth", 1.5),
                alpha=PRETRAINING_STYLES.get(p, {}).get("alpha", 0.8),
                label=p,
            )
            for p in pretraining_vals
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

        # --- Significance annotations ---
        if stats and genotype in stats and "Naive" in pretraining_vals and "Pretrained" in pretraining_vals:
            geno_stats = stats[genotype]
            if ylim is not None:
                _y_data_ceil = ylim[1] / 1.22
            else:
                all_vals = df_gen["distance_mm"].dropna().values
                _y_data_ceil = float(np.percentile(all_vals, 99)) if len(all_vals) else ax.get_ylim()[1]

            bracket_base = _y_data_ceil * 1.03
            bracket_step = _y_data_ceil * 0.06

            for k, tp in enumerate(timepoints):
                if tp not in geno_stats:
                    continue
                sig = geno_stats[tp]["sig"]
                x_naive = box_centres.get((tp, "Naive"), None)
                x_pretrained = box_centres.get((tp, "Pretrained"), None)
                if x_naive is None or x_pretrained is None:
                    continue
                # Stack brackets slightly so overlapping genotype labels don't collide
                bracket_y = bracket_base + k * bracket_step * 0.4
                _draw_significance_bracket(ax, x_naive, x_pretrained, bracket_y, sig, fontsize=11)

        # Apply shared y-axis limits
        if ylim is not None:
            _apply_ylim(ax, ylim)

        plt.tight_layout()

        # Save
        output_path = output_dir / f"distance_timepoints_{genotype}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plot_paths[genotype] = output_path
        print(f"  Saved: {output_path.name}")
        plt.close()

    return plot_paths


def create_distance_timepoint_grid(timepoint_paths, output_dir, time_bins_labels):
    """
    Create a grid layout showing all timepoint plots.
    """
    if len(timepoint_paths) == 0:
        return

    n_cols = 3
    n_rows = (len(timepoint_paths) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 5 * n_rows))

    for idx, (time_label, path) in enumerate(timepoint_paths.items(), 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(time_label, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = output_dir / "distance_timepoint_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid layout: {grid_path.name}")


def create_distance_genotype_grid(genotype_paths, output_dir, genotypes):
    """
    Create a grid layout showing all genotype plots.
    """
    genotypes_with_plots = [g for g in genotypes if g in genotype_paths]

    if len(genotypes_with_plots) == 0:
        return

    n_cols = 3
    n_rows = (len(genotypes_with_plots) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 5 * n_rows))

    for idx, genotype in enumerate(genotypes_with_plots, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        img = plt.imread(genotype_paths[genotype])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(genotype, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = output_dir / "distance_genotype_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid layout: {grid_path.name}")


def validate_60min_values(df_results):
    """
    Compare distance_moved_60min with distance_moved within the summary dataset.
    Both columns are already in the long-format DataFrame as distance_mm and
    distance_moved_mm respectively.
    """
    print("\n" + "=" * 70)
    print("VALIDATION: distance_moved_60min vs distance_moved (within summary)")
    print("=" * 70)

    df_60 = df_results[df_results["timepoint_sec"] == 7200].copy()
    df_60 = df_60.dropna(subset=["distance_mm", "distance_moved_mm"])
    df_60 = df_60[df_60["distance_moved_mm"] > 0]

    if df_60.empty:
        print("  Warning: no 60-min data to validate")
        return

    df_60["pct_diff"] = 100 * (df_60["distance_mm"] - df_60["distance_moved_mm"]) / df_60["distance_moved_mm"]

    print(f"  Flies compared: {len(df_60)}")
    print(f"  Mean % difference:   {df_60['pct_diff'].mean():.4f}%")
    print(f"  Median % difference: {df_60['pct_diff'].median():.4f}%")
    print(f"  Max abs % difference: {df_60['pct_diff'].abs().max():.4f}%")
    print("\n  Sample comparisons (mm):")
    print(
        df_60[["fly", "genotype", "pretraining", "distance_mm", "distance_moved_mm", "pct_diff"]]
        .head(10)
        .to_string(index=False)
    )


def plot_pretraining_difference_over_time(df_results, output_dir):
    """
    For each genotype, compute the Pretrained - Naive difference in median distance
    at each timepoint and plot it across time. Highlights the timepoint of maximum
    absolute difference per genotype.

    Produces:
      - pretraining_difference_over_time.png  (all genotypes on one axes)
      - pretraining_difference_per_genotype.png  (one panel per genotype)
      - pretraining_difference_summary.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Median distance per genotype × pretraining × timepoint
    medians = (
        df_results.groupby(["genotype", "pretraining", "timepoint_sec"])["distance_mm"]
        .median()
        .reset_index()
    )

    pivot = medians.pivot_table(
        index=["genotype", "timepoint_sec"], columns="pretraining", values="distance_mm"
    ).reset_index()

    if "Pretrained" not in pivot.columns or "Naive" not in pivot.columns:
        print("  Warning: Cannot compute pretraining difference — missing Pretrained or Naive condition")
        return

    pivot["difference"] = pivot["Pretrained"] - pivot["Naive"]

    # Build ordered time labels aligned to TIME_BIN_SECS
    timepoint_secs_sorted = sorted(df_results["timepoint_sec"].unique())
    tp_label_map = dict(zip(TIME_BIN_SECS, TIME_BIN_LABELS))
    ordered_labels = [tp_label_map.get(t, str(t)) for t in timepoint_secs_sorted]
    n_tp = len(timepoint_secs_sorted)

    genotypes = sorted(df_results["genotype"].unique())

    # ------------------------------------------------------------------ #
    # Plot 1: All genotypes on one figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        df_gen = pivot[pivot["genotype"] == genotype].sort_values("timepoint_sec")
        if df_gen.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#808080")
        diffs = df_gen["difference"].values
        x = list(range(len(diffs)))

        ax.plot(x, diffs, color=color, label=genotype, linewidth=2, marker="o", markersize=6)

        # Star on max-|diff| timepoint
        if len(diffs) > 0:
            max_idx = int(np.argmax(np.abs(diffs)))
            ax.scatter(max_idx, diffs[max_idx], color=color, s=140, zorder=5,
                       edgecolors="black", linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(n_tp))
    ax.set_xticklabels(ordered_labels)
    ax.set_xlabel("Time point", fontsize=12)
    ax.set_ylabel("Δ Median Distance (mm)\n(Pretrained − Naive)", fontsize=12)
    ax.set_title("Pretrained vs Naive Distance Difference Over Time", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out1 = output_dir / "pretraining_difference_over_time.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1.name}")

    # ------------------------------------------------------------------ #
    # Plot 2: Per-genotype subplots
    # ------------------------------------------------------------------ #
    n_gen = len(genotypes)
    n_cols = min(3, n_gen)
    n_rows = (n_gen + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, genotype in enumerate(genotypes):
        ax = axes_flat[ax_idx]
        df_gen = pivot[pivot["genotype"] == genotype].sort_values("timepoint_sec")
        if df_gen.empty:
            ax.set_visible(False)
            continue

        color = GENOTYPE_COLORS.get(genotype, "#808080")
        diffs = df_gen["difference"].values
        x = list(range(len(diffs)))

        ax.plot(x, diffs, color=color, linewidth=2, marker="o", markersize=6)
        ax.fill_between(x, 0, diffs, alpha=0.15, color=color)

        if len(diffs) > 0:
            max_idx = int(np.argmax(np.abs(diffs)))
            ax.scatter(max_idx, diffs[max_idx], color=color, s=140, zorder=5,
                       edgecolors="black", linewidth=1.5)
            ax.annotate(
                ordered_labels[max_idx],
                (max_idx, diffs[max_idx]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(range(n_tp))
        ax.set_xticklabels(ordered_labels, rotation=45, ha="right")
        ax.set_title(genotype, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx % n_cols == 0:
            ax.set_ylabel("Δ Distance (mm)\n(Pretrained − Naive)", fontsize=10)

    for ax_idx in range(len(genotypes), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("Pretrained vs Naive Distance Difference Over Time", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    out2 = output_dir / "pretraining_difference_per_genotype.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2.name}")

    # ------------------------------------------------------------------ #
    # Console summary & CSV export
    # ------------------------------------------------------------------ #
    print("\n  Maximum Pretrained−Naive difference per genotype:")
    records = []
    for genotype in genotypes:
        df_gen = pivot[pivot["genotype"] == genotype].sort_values("timepoint_sec")
        if df_gen.empty:
            continue
        diffs = df_gen["difference"].values
        tps = df_gen["timepoint_sec"].values
        for i, (t, d) in enumerate(zip(tps, diffs)):
            records.append({
                "genotype": genotype,
                "timepoint_sec": int(t),
                "timepoint_label": tp_label_map.get(t, str(t)),
                "diff_mm": round(float(d), 3),
                "naive_mm": round(float(df_gen["Naive"].values[i]), 3) if "Naive" in df_gen.columns else np.nan,
                "pretrained_mm": round(float(df_gen["Pretrained"].values[i]), 3) if "Pretrained" in df_gen.columns else np.nan,
                "is_max_diff": False,
            })
        max_idx = int(np.argmax(np.abs(diffs)))
        max_label = tp_label_map.get(tps[max_idx], str(tps[max_idx]))
        # Mark in records
        for r in records:
            if r["genotype"] == genotype and r["timepoint_sec"] == int(tps[max_idx]):
                r["is_max_diff"] = True
        print(f"    {genotype:20s}: Δ = {diffs[max_idx]:+.2f} mm  at  {max_label}")

    df_export = pd.DataFrame(records)
    csv_out = output_dir / "pretraining_difference_summary.csv"
    df_export.to_csv(csv_out, index=False)
    print(f"\n  Saved difference table: {csv_out.name}")


def generate_summary_statistics(df_results, output_dir):
    """Generate summary CSV with statistics per genotype, pretraining, and timepoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = (
        df_results.groupby(["genotype", "pretraining", "timepoint"])
        .agg({"distance_mm": ["count", "mean", "std", "median", "min", "max"]})
        .reset_index()
    )

    summary_stats.columns = ["genotype", "pretraining", "timepoint", "n_flies", "mean", "std", "median", "min", "max"]

    output_path = output_dir / "distance_timepoints_summary.csv"
    summary_stats.to_csv(output_path, index=False)
    print(f"\n  Saved summary statistics: {output_path.name}")

    # Also save full results
    full_output = output_dir / "distance_timepoints_full.csv"
    df_results.to_csv(full_output, index=False)
    print(f"  Saved full results: {full_output.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("DISTANCE MOVED AT TIME POINTS ANALYSIS")
    print("=" * 70 + "\n")

    # Load summary and build long-format results DataFrame
    df_results = load_and_prepare()

    if df_results.empty:
        print("\nError: No results generated!")
        return False

    print(f"\nFlies loaded: {df_results['fly'].nunique()}")

    # Validate 60-minute values against distance_moved
    validate_60min_values(df_results)

    # Permutation tests
    print("\n" + "=" * 70)
    print(f"Running permutation tests ({N_PERMUTATIONS} permutations per comparison)...")
    print("=" * 70)
    perm_stats = perform_permutation_tests_at_timepoints(df_results)
    export_permutation_stats(perm_stats, OUTPUT_DIR)

    # Compute a single global y-range shared by all individual plots
    global_ylim = compute_global_ylim(df_results)
    print(f"\n  Global y-range: [{global_ylim[0]:.2f}, {global_ylim[1]:.2f}] mm")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    timepoint_paths = plot_distance_by_timepoint(df_results, OUTPUT_DIR, stats=perm_stats, ylim=global_ylim)
    genotype_paths = plot_distance_by_genotype(df_results, OUTPUT_DIR, stats=perm_stats, ylim=global_ylim)

    create_distance_timepoint_grid(timepoint_paths, OUTPUT_DIR, TIME_BIN_LABELS)
    create_distance_genotype_grid(genotype_paths, OUTPUT_DIR, SELECTED_GENOTYPES)

    print("\n" + "=" * 70)
    print("Generating pretraining difference plots...")
    print("=" * 70)
    plot_pretraining_difference_over_time(df_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Generating summary statistics...")
    print("=" * 70)
    generate_summary_statistics(df_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
