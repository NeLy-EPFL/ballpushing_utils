#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 8d-f: Ball distance trajectories by scent treatment.

This script generates trajectory visualizations showing:
  (d) control (New) versus pre-exposed to fly odors (New + Pre-exposed)
  (e) control (New) versus ethanol washed (Washed)
  (f) control (New) versus washed and pre-exposed (Washed + Pre-exposed)

Data are:
  - Distance the ball was moved relative to its start position (mm)
  - Group averages with bootstrapped 95% CI
  - Binned into 10-min segments (n_bins=6 for a 60-min experiment)
  - Permutation tests with FDR correction (α = 0.05, Benjamini-Hochberg)

Expected results (from published figure):
  - No time bin reached significance after FDR correction for any treatment
  - All FDR-corrected p ≥ 0.11

Usage:
    python edfigure8_def_trajectories.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test: Test mode - use limited data for quick verification
    --n-bins: Number of time bins (default: 6, i.e. 10-min segments for 60-min experiment)
    --n-permutations: Number of permutations (default: 10000)
"""

import argparse
from pathlib import Path

from ballpushing_utils import figure_output_dir, dataset
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Add repository root to path for imports

# Set matplotlib parameters for publication quality
# Pixel to mm conversion (500 px = 30 mm)
PIXELS_PER_MM = 500 / 30

# Fixed color mapping — consistent with metric boxplots
BALLSCENT_COLORS = {
    "New": "#7f7f7f",  # Grey — control
    "Pre-exposed": "#1f77b4",  # Blue  (CtrlScent: pre-exposed to fly odors)
    "Washed": "#2ca02c",  # Green
    "Washed + Pre-exposed": "#ff7f0e",  # Orange
}

# Comparison order: panel (d), (e), (f)
COMPARISONS = [
    ("New", "Pre-exposed", "d"),
    ("New", "Washed", "e"),
    ("New", "Washed + Pre-exposed", "f"),
]

# Dataset path
COORDINATES_PATH = (
    dataset("Ball_scents/Datasets/251103_10_summary_ballscents_Data/coordinates/pooled_coordinates.feather")
)


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


def bootstrap_ci(values, n_bootstrap=1000, seed=42):
    """Bootstrapped 95% CI for the mean."""
    if len(values) <= 1:
        m = float(np.mean(values)) if len(values) == 1 else np.nan
        return m, m
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coordinates_data(coordinates_path=COORDINATES_PATH, test_mode=False):
    """
    Load and preprocess the ball scents coordinates dataset.

    Downsamples to 1 Hz, normalises BallScent labels, and converts
    time from seconds to minutes.

    Parameters
    ----------
    coordinates_path : str or Path
    test_mode : bool   If True limit to 10 flies per condition.

    Returns
    -------
    pd.DataFrame  columns: fly, time (min), distance_ball_0 (pixels), BallScent
    """
    coordinates_path = Path(coordinates_path)
    if not coordinates_path.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coordinates_path}")

    print(f"\n{'='*60}")
    print("LOADING COORDINATES DATASET")
    print(f"{'='*60}")
    print(f"Loading from: {coordinates_path.name}")

    df = pd.read_feather(coordinates_path)
    print(f"✅ Dataset loaded: {df.shape}")

    required_cols = ["time", "distance_ball_0", "fly", "BallScent"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Unique BallScent values (raw): {sorted(df['BallScent'].unique())}")
    print(f"  Total flies: {df['fly'].nunique()}")
    print(f"  Time range: {df['time'].min():.1f}s – {df['time'].max():.1f}s")

    # Normalise BallScent labels
    print("Normalizing BallScent labels...")
    df = normalize_ball_scent_labels(df, group_col="BallScent")

    # Keep only the four conditions shown in the figure
    # Pre-exposed = CtrlScent (new ball pre-exposed to fly odors)
    # Washed + Pre-exposed = Scented (washed then pre-exposed)
    allowed = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]
    df = df[df["BallScent"].isin(allowed)].copy()

    # Downsample to 1 Hz
    print("Downsampling to 1 datapoint per second per fly...")
    df = df.sort_values(["fly", "time"]).copy()
    df["time_rounded"] = df["time"].round(0).astype(int)
    downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg(
        {"distance_ball_0": "mean", "BallScent": "first"}
    )
    downsampled = downsampled.rename(columns={"time_rounded": "time"})

    # Convert time to minutes
    downsampled["time"] = downsampled["time"] / 60.0
    print(f"✅ Downsampled shape: {downsampled.shape}")

    for bs in sorted(downsampled["BallScent"].unique()):
        n_flies = downsampled[downsampled["BallScent"] == bs]["fly"].nunique()
        print(f"  {bs}: {n_flies} flies")

    if test_mode:
        limited_flies = []
        for bs in downsampled["BallScent"].unique():
            flies = downsampled[downsampled["BallScent"] == bs]["fly"].unique()[:10]
            limited_flies.extend(flies)
        downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
        print(f"⚠️  TEST MODE: {downsampled['fly'].nunique()} flies total")

    return downsampled


# ---------------------------------------------------------------------------
# Preprocessing and statistics
# ---------------------------------------------------------------------------


def preprocess_data(
    data, n_bins, time_col="time", value_col="distance_ball_0", group_col="BallScent", subject_col="fly"
):
    """Bin data into time bins and aggregate per fly per bin."""
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    data = data.copy()
    data["time_bin"] = pd.cut(data[time_col], bins=bin_edges, labels=range(n_bins), include_lowest=True).astype(int)

    grouped = data.groupby([group_col, subject_col, "time_bin"])[value_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{value_col}"]
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))
    return grouped


def compute_permutation_test_with_fdr(
    processed_data,
    control_group,
    test_group,
    n_bins,
    metric="avg_distance_ball_0",
    group_col="BallScent",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Permutation tests (mean difference, two-tailed) for each time bin,
    with FDR correction (Benjamini-Hochberg) across all bins.

    Parameters
    ----------
    processed_data : pd.DataFrame   output of preprocess_data
    control_group : str
    test_group : str
    n_bins : int
    metric : str
    group_col : str
    n_permutations : int
    alpha : float
    progress : bool

    Returns
    -------
    dict  per-bin results
    """
    comp_data = processed_data[processed_data[group_col].isin([control_group, test_group])]
    time_bins = sorted(processed_data["time_bin"].unique())

    obs_diffs, p_raw = [], []
    n_ctrl_list, n_test_list = [], []
    mean_ctrl_list, mean_test_list = [], []
    ci_lo_list, ci_hi_list = [], []
    cohens_d_list = []
    bin_c, bin_s, bin_e = [], [], []

    rng = np.random.default_rng(42)
    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    for tb in iterator:
        bin_data = comp_data[comp_data["time_bin"] == tb]
        ctrl_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        if len(bin_data) > 0:
            bin_c.append(float(bin_data["bin_center"].iloc[0]))
            bin_s.append(float(bin_data["bin_start"].iloc[0]))
            bin_e.append(float(bin_data["bin_end"].iloc[0]))
        else:
            bin_c.append(np.nan)
            bin_s.append(np.nan)
            bin_e.append(np.nan)

        n_ctrl_list.append(len(ctrl_vals))
        n_test_list.append(len(test_vals))

        if len(ctrl_vals) == 0 or len(test_vals) == 0:
            obs_diffs.append(np.nan)
            p_raw.append(1.0)
            mean_ctrl_list.append(np.nan)
            mean_test_list.append(np.nan)
            ci_lo_list.append(np.nan)
            ci_hi_list.append(np.nan)
            cohens_d_list.append(np.nan)
            continue

        mean_c = np.mean(ctrl_vals)
        mean_t = np.mean(test_vals)
        obs = mean_t - mean_c
        obs_diffs.append(obs)
        mean_ctrl_list.append(mean_c)
        mean_test_list.append(mean_t)
        cohens_d_list.append(cohens_d(ctrl_vals, test_vals))

        # Bootstrap CI for mean difference
        boot_diffs = np.empty(10000)
        for i in range(10000):
            s1 = rng.choice(ctrl_vals, size=len(ctrl_vals), replace=True)
            s2 = rng.choice(test_vals, size=len(test_vals), replace=True)
            boot_diffs[i] = np.mean(s2) - np.mean(s1)
        ci_lo_list.append(float(np.percentile(boot_diffs, 2.5)))
        ci_hi_list.append(float(np.percentile(boot_diffs, 97.5)))

        # Permutation test
        combined = np.concatenate([ctrl_vals, test_vals])
        n_c = len(ctrl_vals)
        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(combined)
            perm_diffs[i] = np.mean(shuffled[n_c:]) - np.mean(shuffled[:n_c])
        p_raw.append(float(np.mean(np.abs(perm_diffs) >= np.abs(obs))))

    p_raw_arr = np.array(p_raw)
    rejected, p_fdr, _, _ = multipletests(p_raw_arr, alpha=alpha, method="fdr_bh")

    print(f"    Raw significant bins: {int(np.sum(p_raw_arr < alpha))}/{n_bins}")
    print(f"    FDR significant bins: {int(np.sum(rejected))}/{n_bins} (α={alpha})")

    return {
        "time_bins": time_bins,
        "bin_center": bin_c,
        "bin_start": bin_s,
        "bin_end": bin_e,
        "observed_diffs": obs_diffs,
        "n_control": n_ctrl_list,
        "n_test": n_test_list,
        "mean_control": mean_ctrl_list,
        "mean_test": mean_test_list,
        "ci_lower": ci_lo_list,
        "ci_upper": ci_hi_list,
        "cohens_d": cohens_d_list,
        "p_values_raw": p_raw_arr,
        "p_values_fdr": p_fdr,
        "significant_timepoints": np.where(rejected)[0].tolist(),
        "n_significant": int(np.sum(rejected)),
        "n_significant_raw": int(np.sum(p_raw_arr < alpha)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_trajectory_panel(ax, data, control_group, test_group, n_bins, perm_result=None, global_ylim=None):
    """
    Draw a single trajectory comparison panel.

    Shows mean ± bootstrapped 95% CI for control and test group,
    with vertical dotted bin boundaries and red significance markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : pd.DataFrame   full (non-binned) data, time in minutes, distance in pixels
    control_group : str
    test_group : str
    n_bins : int
    perm_result : dict or None
    global_ylim : tuple or None

    Returns
    -------
    float   y_max_data (useful for computing global y range)
    """
    subset = data[data["BallScent"].isin([control_group, test_group])].copy()
    if len(subset) == 0:
        print(f"  Warning: no data for {test_group} vs {control_group}")
        return 0.0

    value_col = "distance_ball_0"
    mm_col = f"{value_col}_mm"
    subset[mm_col] = subset[value_col] / PIXELS_PER_MM

    # First pass: find y_baseline (min of all group means)
    all_means_raw = []
    for group in [control_group, test_group]:
        g_data = subset[subset["BallScent"] == group]
        fly_agg = g_data.groupby(["fly", "time"])[mm_col].mean().reset_index()
        for t in sorted(fly_agg["time"].unique()):
            fly_vals = fly_agg[fly_agg["time"] == t][mm_col].values
            all_means_raw.append(float(np.mean(fly_vals)))
    y_baseline = min(all_means_raw) if all_means_raw else 0.0

    # Second pass: plot with baseline shift
    all_means_shifted = []
    for group in [control_group, test_group]:
        g_data = subset[subset["BallScent"] == group]
        fly_agg = g_data.groupby(["fly", "time"])[mm_col].mean().reset_index()
        time_points = sorted(fly_agg["time"].unique())

        means, ci_lo, ci_hi = [], [], []
        for t in time_points:
            fly_vals = fly_agg[fly_agg["time"] == t][mm_col].values
            m = float(np.mean(fly_vals)) - y_baseline
            means.append(m)
            lo, hi = bootstrap_ci(fly_vals)
            ci_lo.append(lo - y_baseline)
            ci_hi.append(hi - y_baseline)

        all_means_shifted.extend(means)
        color = BALLSCENT_COLORS.get(group, "#333333")
        n_flies = g_data["fly"].nunique()
        ax.plot(
            time_points, means, color=color, linewidth=2.5, linestyle="solid", label=f"{group} (n={n_flies})", zorder=10
        )
        ax.fill_between(time_points, ci_lo, ci_hi, color=color, alpha=0.2, zorder=5)

    y_max_data = max(all_means_shifted) if all_means_shifted else 0.0

    # Vertical dotted bin boundaries
    time_min = subset["time"].min()
    time_max_val = subset["time"].max()
    bin_edges = np.linspace(time_min, time_max_val, n_bins + 1)
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Y-axis limits
    if global_ylim is not None:
        ax.set_ylim(global_ylim)
        y_annot_base = global_ylim[1] * 0.92
    else:
        y_range = y_max_data if y_max_data > 0 else 1.0
        ax.set_ylim(0, y_max_data + 0.08 * y_range)
        y_annot_base = y_max_data + 0.02 * y_range

    # Significance annotations (red asterisks)
    if perm_result is not None:
        for idx in perm_result["significant_timepoints"]:
            p_fdr = perm_result["p_values_fdr"][idx]
            marker = "***" if p_fdr < 0.001 else ("**" if p_fdr < 0.01 else "*")
            bc = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            ax.text(bc, y_annot_base, marker, ha="center", va="center", fontsize=16, fontweight="bold", color="red")

    # Formatting
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Relative ball distance (mm)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return y_max_data


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_trajectory_plots(data, output_dir, n_bins=6, n_permutations=10000, show_progress=True):
    """
    Generate panels (d), (e), (f) of Extended Data Figure 8.

    Produces:
    - One combined PDF with all three panels side by side
    - Individual PDFs for each comparison
    - CSV with per-bin statistics

    Parameters
    ----------
    data : pd.DataFrame   coordinates data (time in minutes, distance in pixels)
    output_dir : Path
    n_bins : int
    n_permutations : int
    show_progress : bool

    Returns
    -------
    dict   per-comparison permutation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING TRAJECTORY PLOTS (panels d-f)")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}  ({60 // n_bins}-min segments for 60-min experiment)")
    print(f"Permutations: {n_permutations}")

    available_scents = set(data["BallScent"].unique())
    comparisons = [(c, t, p) for (c, t, p) in COMPARISONS if c in available_scents and t in available_scents]

    # Bin data
    print(f"\nBinning data into {n_bins} time bins...")
    processed = preprocess_data(data, n_bins=n_bins)
    print(f"Processed shape: {processed.shape}")

    # Run permutation tests
    print(f"\nComputing permutation tests ({n_permutations} permutations each)...")
    perm_results = {}
    for ctrl, test, panel in comparisons:
        print(f"\n  Panel ({panel}): {test} vs {ctrl}:")
        perm_results[(ctrl, test)] = compute_permutation_test_with_fdr(
            processed,
            control_group=ctrl,
            test_group=test,
            n_bins=n_bins,
            metric="avg_distance_ball_0",
            group_col="BallScent",
            n_permutations=n_permutations,
            alpha=0.05,
            progress=show_progress,
        )

    # Compute global y-axis range across all comparisons
    print("\nComputing global y-axis range...")
    global_y_max = 0.0
    for ctrl, test, _ in comparisons:
        subset = data[data["BallScent"].isin([ctrl, test])].copy()
        subset["distance_ball_0_mm"] = subset["distance_ball_0"] / PIXELS_PER_MM
        all_means = []
        for g in [ctrl, test]:
            g_data = subset[subset["BallScent"] == g]
            if len(g_data) > 0:
                all_means.extend(g_data.groupby("time")["distance_ball_0_mm"].mean().values)
        if all_means:
            y_base = min(all_means)
            global_y_max = max(global_y_max, max(all_means) - y_base)

    global_ylim = (0.0, global_y_max + 0.08 * global_y_max)
    print(f"  Global y-axis range: {global_ylim}")

    # ---------- Combined figure (3 panels side by side) ----------
    n_comp = len(comparisons)
    if n_comp > 0:
        fig, axes = plt.subplots(1, n_comp, figsize=(9 * n_comp, 5), sharey=True)
        if n_comp == 1:
            axes = [axes]

        for ax, (ctrl, test, panel) in zip(axes, comparisons):
            plot_trajectory_panel(
                ax,
                data,
                ctrl,
                test,
                n_bins,
                perm_result=perm_results.get((ctrl, test)),
                global_ylim=global_ylim,
            )
            ax.set_title(f"Panel ({panel}): {test}", fontsize=11)

        plt.tight_layout()
        combined_pdf = output_dir / "edfigure8_def_trajectories_combined.pdf"
        fig.savefig(combined_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(combined_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"\n✅ Combined figure saved: {combined_pdf}")
        plt.close(fig)

    # ---------- Individual panels ----------
    for ctrl, test, panel in comparisons:
        fig_ind, ax_ind = plt.subplots(figsize=(9, 5))
        plot_trajectory_panel(
            ax_ind,
            data,
            ctrl,
            test,
            n_bins,
            perm_result=perm_results.get((ctrl, test)),
            global_ylim=global_ylim,
        )
        plt.tight_layout()

        safe_test = test.replace(" + ", "_plus_").replace(" ", "_")
        ind_pdf = output_dir / f"edfigure8{panel}_trajectory_{safe_test}_vs_{ctrl}.pdf"
        fig_ind.savefig(ind_pdf, dpi=300, bbox_inches="tight")
        fig_ind.savefig(ind_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"✅ Individual panel ({panel}) saved: {ind_pdf}")
        plt.close(fig_ind)

    # ---------- Save per-bin statistics CSV ----------
    csv_rows = []
    for ctrl, test, panel in comparisons:
        result = perm_results[(ctrl, test)]
        n_tb = len(result["time_bins"])
        df_comp = pd.DataFrame(
            {
                "panel": panel,
                "comparison": f"{test} vs {ctrl}",
                "condition_control": ctrl,
                "condition_test": test,
                "time_bin": result["time_bins"],
                "bin_start_min": result["bin_start"],
                "bin_end_min": result["bin_end"],
                "bin_center_min": result["bin_center"],
                "n_control": result["n_control"],
                "n_test": result["n_test"],
                "mean_control_mm": np.array(result["mean_control"]) / PIXELS_PER_MM,
                "mean_test_mm": np.array(result["mean_test"]) / PIXELS_PER_MM,
                "mean_diff_mm": np.array(result["observed_diffs"]) / PIXELS_PER_MM,
                "ci_lower_mm": np.array(result["ci_lower"]) / PIXELS_PER_MM,
                "ci_upper_mm": np.array(result["ci_upper"]) / PIXELS_PER_MM,
                "cohens_d": result["cohens_d"],
                "p_value_raw": result["p_values_raw"],
                "p_value_fdr": result["p_values_fdr"],
            }
        )
        df_comp["significant_fdr"] = df_comp["p_value_fdr"] < 0.05
        df_comp["sig_label"] = df_comp["p_value_fdr"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
        csv_rows.append(df_comp)

    if csv_rows:
        stats_csv = output_dir / "edfigure8_def_trajectory_statistics.csv"
        pd.concat(csv_rows, ignore_index=True).to_csv(stats_csv, index=False, float_format="%.6f")
        print(f"✅ Statistics saved: {stats_csv}")

    # Print summary
    print(f"\n{'='*60}")
    print("Statistical Summary (panels d-f):")
    print(f"{'='*60}")
    for ctrl, test, panel in comparisons:
        result = perm_results[(ctrl, test)]
        print(f"\n  Panel ({panel}): {test} vs {ctrl}")
        print(f"    Raw significant bins: {result['n_significant_raw']}/{n_bins}")
        print(f"    FDR significant bins: {result['n_significant']}/{n_bins}")
        if len(result["p_values_fdr"]) > 0:
            min_p = float(np.nanmin(result["p_values_fdr"]))
            print(f"    Min FDR-corrected p: {min_p:.4f}")

    return perm_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main function for Extended Data Figure 8d-f."""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 8d-f: ball distance trajectories by scent treatment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edfigure8_def_trajectories.py                        # Full analysis
  python edfigure8_def_trajectories.py --test                 # Quick test mode
  python edfigure8_def_trajectories.py --n-bins 6             # 10-min segments (default)
  python edfigure8_def_trajectories.py --n-permutations 1000  # Faster (less precise)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data for quick verification")
    parser.add_argument(
        "--n-bins", type=int, default=6, help="Number of time bins (default: 6 = 10-min segments for 60-min experiment)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical tests (default: 10000)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
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
    print("Extended Data Figure 8d-f: Ball Distance Trajectories by Scent")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {args.n_bins} ({60 // args.n_bins}-min segments)")
    print(f"Permutations: {args.n_permutations}")
    if args.test:
        print("⚠️  TEST MODE")

    # Load data
    data = load_coordinates_data(test_mode=args.test)

    # Generate plots and statistics
    perm_results = generate_trajectory_plots(
        data,
        output_dir=output_dir,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        show_progress=not args.no_progress,
    )

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 8d-f generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput files in: {output_dir}")
    print(f"  - edfigure8_def_trajectories_combined.pdf")
    print(f"  - edfigure8d_trajectory_Pre-exposed_vs_New.pdf")
    print(f"  - edfigure8e_trajectory_Washed_vs_New.pdf")
    print(f"  - edfigure8f_trajectory_Washed_plus_Pre-exposed_vs_New.pdf")
    print(f"  - edfigure8_def_trajectory_statistics.csv")
    print(f"\nConditions: New (ctrl), Pre-exposed, Washed, Washed+Pre-exposed")
    print(f"Expected: no significant bins (all FDR-corrected p >= 0.11)")


if __name__ == "__main__":
    main()
