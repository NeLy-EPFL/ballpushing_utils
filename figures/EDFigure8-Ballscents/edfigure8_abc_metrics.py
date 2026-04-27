#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 8a-c: Behavioral metrics by ball scent treatment.

This script generates three boxplots comparing:
  (a) Interaction rate
  (b) Pulling ratio (ratio of pulling / pushing significant interactions)
  (c) Time to first major push (>1.2 mm)

across ball scent treatment groups:
  - New (control, gray)
  - New + Pre-exposed to fly odors (blue)
  - Washed with ethanol (green)
  - Washed + Pre-exposed (orange)

All three treatment groups are compared to control ('New') using pairwise permutation
tests (10,000 permutations), FDR-corrected (α = 0.05, Benjamini-Hochberg) across the
three comparisons within each metric.

Expected results (from published figure):
  - No treatment produced a significant difference in any metric
  - All FDR-corrected p ≥ 0.13

Usage:
    python edfigure8_abc_metrics.py [--test] [--n-permutations N]

Arguments:
    --test: Test mode - use limited data for quick verification
    --n-permutations: Number of permutations (default: 10000)
"""

import argparse
import sys
from pathlib import Path

from ballpushing_utils import figure_output_dir
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Add repository root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set matplotlib parameters for publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Pixel to mm conversion (500 px = 30 mm)
PIXELS_PER_MM = 500 / 30

# Fixed color mapping — consistent with trajectory plots
BALLSCENT_COLORS = {
    "New": "#7f7f7f",  # Grey — control
    "Pre-exposed": "#1f77b4",  # Blue  (CtrlScent: pre-exposed to fly odors)
    "Washed": "#2ca02c",  # Green
    "Washed + Pre-exposed": "#ff7f0e",  # Orange
}

# Fixed display order: control first, then others
FIXED_ORDER = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]

# Metrics for panels (a), (b), (c) — columns in summary feather
PANEL_METRICS = [
    {
        "col": "overall_interaction_rate",
        "label": "Interaction rate",
        "unit": "",
        "panel": "a",
        "convert": None,  # No unit conversion needed
    },
    {
        "col": "pulling_ratio",
        "label": "Pulling ratio",
        "unit": "",
        "panel": "b",
        "convert": None,
    },
    {
        "col": "first_major_event_time",
        "label": "Time to first major push (min)",
        "unit": "min",
        "panel": "c",
        "convert": lambda x: x / 60.0,  # seconds → minutes
    },
]

# Dataset paths
SUMMARY_PATH = (
    "/mnt/upramdya_data/MD/Ball_scents/Datasets/" "251103_10_summary_ballscents_Data/summary/pooled_summary.feather"
)
CTRL_SUMMARY_PATHS = [
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
    "250815_18_summary_control_folders_Data/summary/230704_FeedingState_1_AM_Videos_Tracked_summary.feather",
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
    "250815_18_summary_control_folders_Data/summary/230705_FeedingState_2_AM_Videos_Tracked_summary.feather",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cohens_d(group1, group2):
    """Cohen's d effect size (group2 − group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def permutation_test(group1, group2, n_permutations=10000, rng=None):
    """Two-tailed permutation test on median difference."""
    if len(group1) == 0 or len(group2) == 0:
        return 1.0
    if rng is None:
        rng = np.random.default_rng(42)
    obs_stat = float(np.median(group2) - np.median(group1))
    combined = np.concatenate([group1, group2])
    n_ctrl = len(group1)
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_diffs[i] = np.median(shuffled[n_ctrl:]) - np.median(shuffled[:n_ctrl])
    return float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))


def normalize_ball_scent_labels(df, group_col="BallScent"):
    """Normalize BallScent values to canonical display labels."""
    from difflib import get_close_matches

    design_keys = ["Ctrl", "CtrlScent", "Washed", "Scented", "New", "NewScent"]
    available = pd.Series(df[group_col].dropna().unique()).astype(str).tolist()

    mapping = {}
    for val in available:
        if val in design_keys:
            mapping[val] = val
            continue
        vs = str(val).strip().lower()
        found = None
        for k in design_keys:
            ks = k.lower()
            if vs == ks or ks in vs or vs in ks:
                found = k
                break
        if not found:
            candidates = get_close_matches(vs, [k.lower() for k in design_keys], n=1, cutoff=0.6)
            if candidates:
                for k in design_keys:
                    if k.lower() == candidates[0]:
                        found = k
                        break
        mapping[val] = found if found is not None else val

    for val in list(mapping.keys()):
        if mapping[val] == val and val not in design_keys:
            vs = str(val).strip().lower()
            if "new" in vs and "scent" in vs:
                mapping[val] = "NewScent"
            elif "new" in vs:
                mapping[val] = "New"
            elif "wash" in vs and "scent" in vs:
                mapping[val] = "Scented"
            elif "wash" in vs:
                mapping[val] = "Washed"
            elif "ctrl" in vs and "scent" in vs:
                mapping[val] = "CtrlScent"
            elif "scent" in vs or "pre" in vs or "exposed" in vs:
                mapping[val] = "CtrlScent"

    display_map = {
        "Ctrl": "Ctrl",
        "CtrlScent": "Pre-exposed",
        "Washed": "Washed",
        "Scented": "Washed + Pre-exposed",
        "New": "New",
        "NewScent": "New + Pre-exposed",
    }

    df = df.copy()
    df[group_col] = df[group_col].map(lambda x: display_map.get(mapping.get(str(x), str(x)), str(x)))
    return df


def load_dataset(test_mode=False):
    """Load and clean the ball scents summary dataset."""
    print(f"\n{'='*60}")
    print("LOADING SUMMARY DATASET")
    print(f"{'='*60}")

    try:
        dataset = pd.read_feather(SUMMARY_PATH)
        print(f"✅ Loaded: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {SUMMARY_PATH}")

    # Drop TNT-specific metadata columns
    tnt_cols = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Genotype",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
    ]
    drop_cols = [c for c in tnt_cols if c in dataset.columns]
    if drop_cols:
        dataset = dataset.drop(columns=drop_cols)

    # Convert bool columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns
    for col in ["insight_effect", "insight_effect_log", "exit_time"]:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])

    # Append historical Ctrl cohorts
    ctrl_dfs = []
    for path in CTRL_SUMMARY_PATHS:
        try:
            ctrl_df = pd.read_feather(path)
            if "FeedingState" in ctrl_df.columns:
                ctrl_df = ctrl_df[ctrl_df["FeedingState"] == "starved_noWater"].copy()
                if len(ctrl_df) == 0:
                    continue
            ctrl_df["BallScent"] = "Ctrl"
            ctrl_dfs.append(ctrl_df)
            print(f"  Loaded Ctrl cohort: {ctrl_df.shape}")
        except FileNotFoundError:
            print(f"  ⚠️  Not found: {Path(path).name}")

    if ctrl_dfs:
        dataset = pd.concat([dataset] + ctrl_dfs, ignore_index=True, sort=False)
        drop_cols2 = [c for c in tnt_cols if c in dataset.columns]
        if drop_cols2:
            dataset = dataset.drop(columns=drop_cols2)
        for col in dataset.columns:
            if dataset[col].dtype == bool:
                dataset[col] = dataset[col].astype(int)
        print(f"  Combined dataset: {dataset.shape}")

    print(f"\nNormalizing BallScent labels...")
    dataset = normalize_ball_scent_labels(dataset, group_col="BallScent")

    # Keep only the four conditions shown in the figure
    # Pre-exposed = CtrlScent (new ball pre-exposed to fly odors)
    # Washed + Pre-exposed = Scented (washed then pre-exposed)
    allowed = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]
    dataset = dataset[dataset["BallScent"].isin(allowed)].copy()

    print(f"Ball scents in dataset: {sorted(dataset['BallScent'].unique())}")
    for bs in FIXED_ORDER:  # noqa: E501
        if bs in dataset["BallScent"].values:
            n = len(dataset[dataset["BallScent"] == bs])
            print(f"  {bs}: n={n}")

    if test_mode:
        sampled = []
        for bs in allowed:
            part = dataset[dataset["BallScent"] == bs]
            sampled.append(part.head(min(30, len(part))))
        dataset = pd.concat(sampled, ignore_index=True)
        print(f"⚠️  TEST MODE: {dataset.shape}")

    return dataset


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def run_permutation_tests(data, metric_col, control="New", n_permutations=10000):
    """
    Run pairwise permutation tests (control vs each treatment) for one metric.

    FDR correction is applied across the 3 comparisons.

    Returns a list of result dicts.
    """
    rng = np.random.default_rng(42)
    order = [g for g in FIXED_ORDER if g != control and g in data["BallScent"].values]

    ctrl_vals = data[data["BallScent"] == control][metric_col].dropna().values

    raw_pvals = []
    results = []

    for group in order:
        test_vals = data[data["BallScent"] == group][metric_col].dropna().values
        p_raw = permutation_test(ctrl_vals, test_vals, n_permutations=n_permutations, rng=rng)
        d = cohens_d(ctrl_vals, test_vals)
        raw_pvals.append(p_raw)
        results.append(
            {
                "metric": metric_col,
                "comparison": f"{group} vs {control}",
                "group": group,
                "control": control,
                "n_test": int(len(test_vals)),
                "n_control": int(len(ctrl_vals)),
                "median_test": float(np.median(test_vals)) if len(test_vals) else np.nan,
                "median_control": float(np.median(ctrl_vals)) if len(ctrl_vals) else np.nan,
                "mean_test": float(np.mean(test_vals)) if len(test_vals) else np.nan,
                "mean_control": float(np.mean(ctrl_vals)) if len(ctrl_vals) else np.nan,
                "p_value_raw": p_raw,
                "cohens_d": d,
                "n_permutations": n_permutations,
            }
        )
        print(f"    {group} vs {control}: p_raw={p_raw:.4f}, d={d:.3f}")

    # FDR correction (Benjamini-Hochberg)
    rejected, pvals_fdr, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    for i, r in enumerate(results):
        r["p_value_fdr"] = float(pvals_fdr[i])
        r["significant_fdr"] = bool(rejected[i])
        p = pvals_fdr[i]
        r["sig_level"] = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {r['comparison']}: p_fdr={p:.4f} ({r['sig_level']})")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metric_panel(ax, data, metric_col, y_label, convert_fn, results, control="New"):
    """
    Draw a single boxplot panel for one metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : pd.DataFrame
    metric_col : str
    y_label : str
    convert_fn : callable or None   unit conversion applied to raw values
    results : list of dicts         permutation test results for this metric
    control : str
    """
    order = [g for g in FIXED_ORDER if g in data["BallScent"].values]
    colors = [BALLSCENT_COLORS[g] for g in order]

    # Prepare values (with optional unit conversion)
    group_vals = {}
    for g in order:
        vals = data[data["BallScent"] == g][metric_col].dropna().values
        if convert_fn is not None:
            vals = convert_fn(vals)
        # Clip outliers at 99th percentile for better visualisation
        if len(vals) > 0:
            clip_val = np.nanpercentile(vals, 99)
            vals = np.clip(vals, None, clip_val)
        group_vals[g] = vals

    x_positions = list(range(len(order)))

    # Boxplot
    bp = ax.boxplot(
        [group_vals[g] for g in order],
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for i, g in enumerate(order):
        vals = group_vals[g]
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(i + jitter, vals, s=20, color=BALLSCENT_COLORS[g], alpha=0.5, edgecolors="none", zorder=3)

    # Significance brackets
    control_idx = order.index(control)
    all_vals = np.concatenate([v for v in group_vals.values() if len(v) > 0])
    y_max = float(np.max(all_vals))
    y_min = float(np.min(all_vals))
    y_range = y_max - y_min if y_max > y_min else 1.0

    bracket_base = y_max + 0.08 * y_range
    bracket_step = 0.12 * y_range

    sig_results = [r for r in results if r["sig_level"] != "ns"]
    for bracket_num, r in enumerate(sig_results):
        test_idx = order.index(r["group"])
        x1, x2 = sorted([control_idx, test_idx])
        h = bracket_base + bracket_num * bracket_step
        ax.plot([x1, x1], [h - 0.01 * y_range, h], "k-", linewidth=1.5)
        ax.plot([x1, x2], [h, h], "k-", linewidth=1.5)
        ax.plot([x2, x2], [h - 0.01 * y_range, h], "k-", linewidth=1.5)
        ax.text(
            (x1 + x2) / 2,
            h + 0.01 * y_range,
            r["sig_level"],
            ha="center",
            va="bottom",
            fontsize=12,
            color="red",
            fontweight="bold",
        )

    n_sig = len(sig_results)
    y_top = bracket_base + max(n_sig, 1) * bracket_step + 0.08 * y_range
    ax.set_ylim(y_min - 0.05 * y_range, y_top)

    # X-axis labels with n
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{g}\n(n={len(group_vals[g])})" for g in order],
        fontsize=8,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(y_label, fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_metric_plots(data, output_dir, n_permutations=10000):
    """
    Generate panels (a), (b), (c) of Extended Data Figure 8.

    Produces:
    - One combined PDF with all three panels side by side
    - Individual PDFs for each panel
    - CSV with all statistical results

    Returns
    -------
    pd.DataFrame  statistics table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    control = "New"

    all_stats = []

    # ---------- Run all permutation tests first ----------
    metric_results = {}
    for m in PANEL_METRICS:
        col = m["col"]
        if col not in data.columns:
            print(f"  ⚠️  Column '{col}' not found — skipping panel ({m['panel']})")
            metric_results[col] = []
            continue
        print(f"\nPermutation tests for '{col}' (panel {m['panel']}):")
        res = run_permutation_tests(data, col, control=control, n_permutations=n_permutations)
        metric_results[col] = res
        all_stats.extend(res)

    # ---------- Combined figure (3 panels side by side) ----------
    available = [m for m in PANEL_METRICS if m["col"] in data.columns]
    n_panels = len(available)

    if n_panels > 0:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        for ax, m in zip(axes, available):
            col = m["col"]
            res = metric_results[col]
            # Apply conversion to data for plotting
            plot_df = data.copy()
            if m["convert"] is not None:
                plot_df[col] = m["convert"](plot_df[col])
            plot_metric_panel(ax, plot_df, col, m["label"], None, res, control=control)

        plt.tight_layout()
        combined_pdf = output_dir / "edfigure8_abc_metrics_combined.pdf"
        fig.savefig(combined_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(combined_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"\n✅ Combined figure saved: {combined_pdf}")
        plt.close(fig)

    # ---------- Individual panels ----------
    for m in available:
        col = m["col"]
        res = metric_results[col]
        plot_df = data.copy()
        if m["convert"] is not None:
            plot_df[col] = m["convert"](plot_df[col])

        fig_ind, ax_ind = plt.subplots(figsize=(4, 6))
        plot_metric_panel(ax_ind, plot_df, col, m["label"], None, res, control=control)
        plt.tight_layout()

        ind_pdf = output_dir / f"edfigure8{m['panel']}_{col}.pdf"
        fig_ind.savefig(ind_pdf, dpi=300, bbox_inches="tight")
        fig_ind.savefig(ind_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"✅ Individual panel ({m['panel']}) saved: {ind_pdf}")
        plt.close(fig_ind)

    # ---------- Save statistics ----------
    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_file = output_dir / "edfigure8_abc_statistics.csv"
        stats_df.to_csv(stats_file, index=False, float_format="%.6f")
        print(f"✅ Statistics saved: {stats_file}")

    return stats_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main function for Extended Data Figure 8a-c."""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 8a-c: behavioral metrics by ball scent treatment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edfigure8_abc_metrics.py                        # Full analysis
  python edfigure8_abc_metrics.py --test                 # Quick test mode
  python edfigure8_abc_metrics.py --n-permutations 1000  # Fewer permutations
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data for quick verification")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical tests (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <figures_root>/EDFigure8/<script_stem>/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else figure_output_dir("EDFigure8", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 8a-c: Behavioral Metrics by Ball Scent")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Permutations: {args.n_permutations}")
    if args.test:
        print("⚠️  TEST MODE")

    # Load data
    data = load_dataset(test_mode=args.test)

    # Generate plots and statistics
    stats_df = generate_metric_plots(data, output_dir, n_permutations=args.n_permutations)

    # Print summary
    print(f"\n{'='*60}")
    print("Statistical Summary (panels a-c):")
    print(f"{'='*60}")
    for _, row in stats_df.iterrows():
        print(f"\n  {row['comparison']} ({row['metric']}):")
        print(f"    control median = {row['median_control']:.4f}  (n={row['n_control']})")
        print(f"    test    median = {row['median_test']:.4f}  (n={row['n_test']})")
        print(f"    p_raw={row['p_value_raw']:.4f},  p_fdr={row['p_value_fdr']:.4f}  ({row['sig_level']})")

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 8a-c generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput files in: {output_dir}")
    print(f"  - edfigure8_abc_metrics_combined.pdf      (all three panels)")
    print(f"  - edfigure8a_overall_interaction_rate.pdf  (a) interaction rate")
    print(f"  - edfigure8b_pulling_ratio.pdf             (b) pulling ratio")
    print(f"  - edfigure8c_first_major_event_time.pdf    (c) time to first major push")
    print(f"  - edfigure8_abc_statistics.csv             statistics")
    print(f"\nConditions: New (ctrl), Pre-exposed, Washed, Washed+Pre-exposed")
    print(f"Expected: all FDR-corrected p >= 0.13 (no significant differences)")


if __name__ == "__main__":
    main()
