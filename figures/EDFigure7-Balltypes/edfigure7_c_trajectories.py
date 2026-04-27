#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 7c: Ball distance trajectories by surface treatment.

This script generates trajectory visualizations showing:
- Distance the ball was moved relative to the start position over time
- Two panels: control vs rusty (top) and control vs sandpaper (bottom)
- Data binned into 5-min segments (n_bins=12 for a 60-min experiment)
- Group averages with bootstrapped 95% CI
- Permutation tests with FDR correction (α = 0.05)

Expected results (from published figure):
- No time bins reached significance after FDR correction for either treatment
- All FDR-corrected p ≥ 0.08

Usage:
    python edfigure7_c_trajectories.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test: Test mode - use limited data for quick verification
    --n-bins: Number of time bins (default: 12, i.e. 5-min segments for 60-min experiment)
    --n-permutations: Number of permutations (default: 10000)
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Add repository root to path for imports

from ballpushing_utils import figure_output_dir, dataset
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# Set matplotlib parameters for publication quality
# Pixel to mm conversion (500 px = 30 mm)
PIXELS_PER_MM = 500 / 30

# Fixed color mapping consistent with panel (b) and original scripts
BALLTYPE_COLORS = {
    "control": "#7f7f7f",  # Grey
    "rusty": "#d62728",  # Red
    "sandpaper": "#ff69b4",  # Pink
}

# Dataset path
COORDINATES_PATH = (
    dataset("Ballpushing_Balltypes/Datasets/250815_17_summary_ball_types_Data/coordinates/pooled_coordinates.feather")
)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def bootstrap_ci(values, n_bootstrap=1000):
    """
    Compute bootstrapped 95% CI for the mean of an array.

    Parameters:
    -----------
    values : array-like
        Values to bootstrap
    n_bootstrap : int
        Number of bootstrap samples

    Returns:
    --------
    tuple (lower, upper)
    """
    if len(values) <= 1:
        m = float(np.mean(values)) if len(values) == 1 else np.nan
        return m, m
    rng = np.random.default_rng(42)
    boots = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_coordinates_data(coordinates_path=COORDINATES_PATH, test_mode=False):
    """
    Load and preprocess the ball types coordinates dataset.

    Applies downsampling to 1 datapoint per second per fly,
    renames ball types, and converts time to minutes.

    Parameters:
    -----------
    coordinates_path : str or Path
        Path to the pooled coordinates feather file
    test_mode : bool
        If True, limit to 10 flies per condition for quick verification

    Returns:
    --------
    pd.DataFrame
        Preprocessed coordinates with columns: fly, time (min), distance_ball_0, BallType
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

    required_cols = ["time", "distance_ball_0", "fly", "BallType"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Rename ball types for clarity (matching original scripts)
    balltype_mapping = {"ctrl": "control", "sand": "sandpaper"}
    df["BallType"] = df["BallType"].map(balltype_mapping).fillna(df["BallType"])

    print(f"  Unique ball types: {sorted(df['BallType'].unique())}")
    print(f"  Total flies: {df['fly'].nunique()}")
    print(f"  Time range: {df['time'].min():.1f}s – {df['time'].max():.1f}s")

    # Downsample to 1 Hz by rounding to nearest integer second
    print("Downsampling to 1 datapoint per second per fly...")
    df = df.sort_values(["fly", "time"]).copy()
    df["time_rounded"] = df["time"].round(0).astype(int)

    downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg(
        {"distance_ball_0": "mean", "BallType": "first"}
    )
    downsampled = downsampled.rename(columns={"time_rounded": "time"})

    # Convert time to minutes
    downsampled["time"] = downsampled["time"] / 60.0

    print(f"✅ Downsampled shape: {downsampled.shape}")
    for bt in sorted(downsampled["BallType"].unique()):
        n_flies = downsampled[downsampled["BallType"] == bt]["fly"].nunique()
        print(f"  {bt}: {n_flies} flies")

    if test_mode:
        limited_flies = []
        for bt in downsampled["BallType"].unique():
            bt_flies = downsampled[downsampled["BallType"] == bt]["fly"].unique()[:10]
            limited_flies.extend(bt_flies)
        downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
        print(f"⚠️  TEST MODE: Limited to {downsampled['fly'].nunique()} flies total")

    return downsampled


def preprocess_data(
    data, n_bins=12, time_col="time", value_col="distance_ball_0", group_col="BallType", subject_col="fly"
):
    """
    Bin data into time bins and aggregate per fly per bin.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw trajectory data (time in minutes)
    n_bins : int
        Number of time bins
    time_col : str
        Column name for time (minutes)
    value_col : str
        Column name for distance (pixels)
    group_col : str
        Column name for grouping (BallType)
    subject_col : str
        Column name for individual subjects (fly)

    Returns:
    --------
    pd.DataFrame
        Aggregated data with columns: group, subject, time_bin,
        avg_{value_col}, bin_center, bin_start, bin_end
    """
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
    group_col="BallType",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Run permutation tests for each time bin and apply FDR correction.

    Uses mean difference as the test statistic (two-tailed).

    Parameters:
    -----------
    processed_data : pd.DataFrame
        Binned data with per-fly-per-bin averages
    control_group : str
        Name of the control group
    test_group : str
        Name of the test group
    n_bins : int
        Number of time bins
    metric : str
        Column name for the metric to test
    group_col : str
        Column name for grouping
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level for FDR correction
    progress : bool
        Show progress bar

    Returns:
    --------
    dict
        Per-bin results including observed diffs, raw p-values, FDR-corrected p-values,
        bootstrap CIs, and Cohen's d
    """
    comparison_data = processed_data[processed_data[group_col].isin([control_group, test_group])]
    time_bins = sorted(processed_data["time_bin"].unique())

    observed_diffs, p_values_raw = [], []
    n_control_vals, n_test_vals = [], []
    mean_control_vals, mean_test_vals = [], []
    ci_lower_vals, ci_upper_vals = [], []
    cohens_d_vals = []
    bin_center_vals, bin_start_vals, bin_end_vals = [], [], []

    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    rng = np.random.default_rng(42)

    for time_bin in iterator:
        bin_data = comparison_data[comparison_data["time_bin"] == time_bin]
        control_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        # Store bin position
        if len(bin_data) > 0:
            bin_center_vals.append(float(bin_data["bin_center"].iloc[0]))
            bin_start_vals.append(float(bin_data["bin_start"].iloc[0]))
            bin_end_vals.append(float(bin_data["bin_end"].iloc[0]))
        else:
            bin_center_vals.append(np.nan)
            bin_start_vals.append(np.nan)
            bin_end_vals.append(np.nan)

        n_control_vals.append(len(control_vals))
        n_test_vals.append(len(test_vals))

        if len(control_vals) == 0 or len(test_vals) == 0:
            observed_diffs.append(np.nan)
            p_values_raw.append(1.0)
            mean_control_vals.append(np.nan)
            mean_test_vals.append(np.nan)
            ci_lower_vals.append(np.nan)
            ci_upper_vals.append(np.nan)
            cohens_d_vals.append(np.nan)
            continue

        mean_ctrl = np.mean(control_vals)
        mean_tst = np.mean(test_vals)
        obs_diff = mean_tst - mean_ctrl
        observed_diffs.append(obs_diff)
        mean_control_vals.append(mean_ctrl)
        mean_test_vals.append(mean_tst)

        # Bootstrap CI for the mean difference
        n_bootstrap = 10000
        diffs_boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            s1 = rng.choice(control_vals, size=len(control_vals), replace=True)
            s2 = rng.choice(test_vals, size=len(test_vals), replace=True)
            diffs_boot[i] = np.mean(s2) - np.mean(s1)
        ci_lower_vals.append(float(np.percentile(diffs_boot, 2.5)))
        ci_upper_vals.append(float(np.percentile(diffs_boot, 97.5)))

        cohens_d_vals.append(cohens_d(control_vals, test_vals))

        # Permutation test (mean difference, two-tailed)
        combined = np.concatenate([control_vals, test_vals])
        n_ctrl = len(control_vals)
        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(combined)
            perm_diffs[i] = np.mean(shuffled[n_ctrl:]) - np.mean(shuffled[:n_ctrl])
        p_values_raw.append(float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))))

    # FDR correction across all time bins
    p_values_raw = np.array(p_values_raw)
    rejected, p_values_fdr, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

    print(f"    Raw significant bins: {np.sum(p_values_raw < alpha)}/{n_bins}")
    print(f"    FDR significant bins: {np.sum(rejected)}/{n_bins} (α={alpha})")

    return {
        "time_bins": time_bins,
        "bin_center": bin_center_vals,
        "bin_start": bin_start_vals,
        "bin_end": bin_end_vals,
        "observed_diffs": observed_diffs,
        "n_control": n_control_vals,
        "n_test": n_test_vals,
        "mean_control": mean_control_vals,
        "mean_test": mean_test_vals,
        "ci_lower": ci_lower_vals,
        "ci_upper": ci_upper_vals,
        "cohens_d": cohens_d_vals,
        "p_values_raw": p_values_raw,
        "p_values_fdr": p_values_fdr,
        "significant_timepoints": np.where(rejected)[0].tolist(),
        "n_significant": int(np.sum(rejected)),
        "n_significant_raw": int(np.sum(p_values_raw < alpha)),
    }


def plot_trajectory_panel(ax, data, control_group, test_group, n_bins, perm_result=None, global_ylim=None):
    """
    Draw a single trajectory comparison panel on the given axis.

    Shows mean ± bootstrapped 95% CI for each group over time,
    with vertical dotted lines marking bin boundaries and
    red significance asterisks for FDR-significant bins.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    data : pd.DataFrame
        Full (non-binned) trajectory data in minutes, distance in pixels
    control_group : str
        Name of the control group
    test_group : str
        Name of the test group
    n_bins : int
        Number of time bins
    perm_result : dict, optional
        Output of compute_permutation_test_with_fdr for this comparison
    global_ylim : tuple, optional
        (y_min, y_max) to apply uniformly across panels

    Returns:
    --------
    float
        The y_max_data value (before global_ylim override), useful for range calc
    """
    subset = data[data["BallType"].isin([control_group, test_group])].copy()
    if len(subset) == 0:
        print(f"  Warning: No data for {test_group} vs {control_group}")
        return 0.0

    # Convert pixels to mm
    value_col = "distance_ball_0"
    value_col_mm = f"{value_col}_mm"
    subset[value_col_mm] = subset[value_col] / PIXELS_PER_MM

    # Compute per-fly-per-timepoint mean, then grand mean + CI across flies
    all_group_means = []
    for group in [control_group, test_group]:
        g_data = subset[subset["BallType"] == group]
        fly_agg = g_data.groupby(["fly", "time"])[value_col_mm].mean().reset_index()
        time_points = sorted(fly_agg["time"].unique())

        means, ci_lo, ci_hi = [], [], []
        for t in time_points:
            fly_vals = fly_agg[fly_agg["time"] == t][value_col_mm].values
            m = float(np.mean(fly_vals))
            means.append(m)
            lo, hi = bootstrap_ci(fly_vals)
            ci_lo.append(lo)
            ci_hi.append(hi)

        all_group_means.extend(means)

        color = BALLTYPE_COLORS.get(group, "#333333")
        n_flies = g_data["fly"].nunique()
        label = f"{group} (n={n_flies})"

        ax.plot(time_points, means, color=color, linewidth=2.5, linestyle="solid", label=label, zorder=10)
        ax.fill_between(time_points, ci_lo, ci_hi, color=color, alpha=0.2, zorder=5)

    # Baseline shift: shift y so the minimum mean starts at 0
    y_baseline = min(all_group_means)

    # Replot with baseline shift — recompute shifted coords from the already-plotted lines
    # More cleanly: redo the loop shifting values
    ax.cla()  # clear and redo with shift

    all_group_means_shifted = []
    for group in [control_group, test_group]:
        g_data = subset[subset["BallType"] == group]
        fly_agg = g_data.groupby(["fly", "time"])[value_col_mm].mean().reset_index()
        time_points = sorted(fly_agg["time"].unique())

        means, ci_lo, ci_hi = [], [], []
        for t in time_points:
            fly_vals = fly_agg[fly_agg["time"] == t][value_col_mm].values
            m = float(np.mean(fly_vals)) - y_baseline
            means.append(m)
            lo, hi = bootstrap_ci(fly_vals)
            ci_lo.append(lo - y_baseline)
            ci_hi.append(hi - y_baseline)

        all_group_means_shifted.extend(means)

        color = BALLTYPE_COLORS.get(group, "#333333")
        n_flies = g_data["fly"].nunique()
        label = f"{group} (n={n_flies})"

        ax.plot(time_points, means, color=color, linewidth=2.5, linestyle="solid", label=label, zorder=10)
        ax.fill_between(time_points, ci_lo, ci_hi, color=color, alpha=0.2, zorder=5)

    y_max_data = max(all_group_means_shifted) if all_group_means_shifted else 0.0

    # Draw vertical dotted lines for bin boundaries
    time_min = subset["time"].min()
    time_max = subset["time"].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Set y-axis limits
    if global_ylim is not None:
        ax.set_ylim(global_ylim)
        y_range_for_annot = global_ylim[1]
        y_annot_base = global_ylim[1] * 0.92
    else:
        y_range = y_max_data if y_max_data > 0 else 1.0
        ax.set_ylim(0, y_max_data + 0.08 * y_range)
        y_annot_base = y_max_data + 0.02 * y_range
        y_range_for_annot = y_max_data

    # Significance annotations
    if perm_result is not None:
        for idx in perm_result["significant_timepoints"]:
            p_fdr = perm_result["p_values_fdr"][idx]
            if p_fdr < 0.001:
                marker = "***"
            elif p_fdr < 0.01:
                marker = "**"
            elif p_fdr < 0.05:
                marker = "*"
            else:
                continue
            bin_center = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            ax.text(
                bin_center, y_annot_base, marker, ha="center", va="center", fontsize=16, fontweight="bold", color="red"
            )

    # Axis formatting
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Relative ball distance (mm)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return y_max_data


def generate_trajectory_plots(
    data, control_condition="control", n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate Extended Data Figure 7c trajectory plots.

    Creates:
    - Individual PDF per comparison (rusty vs control, sandpaper vs control)
    - Combined figure with both comparisons stacked vertically
    - CSV with per-bin statistical results

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed coordinates (time in minutes, distance in pixels, BallType column)
    control_condition : str
        Name of control group
    n_bins : int
        Number of time bins (12 gives 5-min bins for a 60-min experiment)
    n_permutations : int
        Number of permutations for statistical testing
    output_dir : Path or str
        Directory to save outputs
    show_progress : bool
        Show tqdm progress bars
    """
    output_dir = Path(output_dir) if output_dir is not None else figure_output_dir("EDFigure7", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING TRAJECTORY PLOTS (panel c)")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins} (5-min segments for 60-min experiment)")
    print(f"Permutations: {n_permutations}")
    print(f"Control: {control_condition}")

    # Only plot rusty and sandpaper vs control (as per figure legend)
    ball_types = sorted(data["BallType"].unique())
    test_conditions = [b for b in ball_types if b != control_condition and b in ["rusty", "sandpaper"]]
    print(f"Test conditions: {test_conditions}")

    if control_condition not in ball_types:
        raise ValueError(f"Control condition '{control_condition}' not found in data")

    # Preprocess (bin per fly)
    print(f"\nBinning data into {n_bins} time bins...")
    processed = preprocess_data(data, n_bins=n_bins)
    print(f"Processed shape: {processed.shape}")

    # Run permutation tests for each comparison
    print(f"\nComputing permutation tests ({n_permutations} permutations each)...")
    perm_results = {}
    for test_cond in test_conditions:
        print(f"\n  {test_cond} vs {control_condition}:")
        perm_results[test_cond] = compute_permutation_test_with_fdr(
            processed,
            control_group=control_condition,
            test_group=test_cond,
            n_bins=n_bins,
            metric="avg_distance_ball_0",
            group_col="BallType",
            n_permutations=n_permutations,
            alpha=0.05,
            progress=show_progress,
        )

    # Determine global y-axis range across all comparisons
    print("\nComputing global y-axis range...")
    global_y_max = 0.0
    for test_cond in test_conditions:
        subset = data[data["BallType"].isin([control_condition, test_cond])].copy()
        subset["distance_ball_0_mm"] = subset["distance_ball_0"] / PIXELS_PER_MM
        all_means = []
        for g in [control_condition, test_cond]:
            g_data = subset[subset["BallType"] == g]
            if len(g_data) > 0:
                t_means = g_data.groupby("time")["distance_ball_0_mm"].mean().values
                all_means.extend(t_means)
        if all_means:
            y_baseline = min(all_means)
            y_max_shifted = max(all_means) - y_baseline
            global_y_max = max(global_y_max, y_max_shifted)

    global_ylim = (0.0, global_y_max + 0.08 * global_y_max)
    print(f"  Global y-axis range: {global_ylim}")

    # Individual plots
    # The figure legend orders: rusty (top), sandpaper (bottom) → sort for consistency
    ordered_test_conds = sorted(test_conditions)

    for test_cond in ordered_test_conds:
        print(f"\nCreating individual plot: {test_cond} vs {control_condition}...")
        fig_single, ax_single = plt.subplots(figsize=(10, 5))
        plot_trajectory_panel(
            ax_single,
            data,
            control_condition,
            test_cond,
            n_bins,
            perm_result=perm_results.get(test_cond),
            global_ylim=global_ylim,
        )
        fig_single.suptitle(f"Control vs {test_cond}", fontsize=14, y=1.01)
        plt.tight_layout()

        out_pdf = output_dir / f"edfigure7c_trajectory_{test_cond}_vs_{control_condition}.pdf"
        fig_single.savefig(out_pdf, dpi=300, bbox_inches="tight")
        fig_single.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"  ✅ Saved: {out_pdf}")
        plt.close(fig_single)

    # Save CSV with detailed per-bin statistics
    csv_rows = []
    for test_cond in ordered_test_conds:
        result = perm_results[test_cond]
        n_time_bins = len(result["time_bins"])
        df_comp = pd.DataFrame(
            {
                "comparison": f"{test_cond} vs {control_condition}",
                "condition_control": control_condition,
                "condition_test": test_cond,
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
        stats_csv = output_dir / "edfigure7c_trajectory_statistics.csv"
        pd.concat(csv_rows, ignore_index=True).to_csv(stats_csv, index=False, float_format="%.6f")
        print(f"✅ Statistics saved: {stats_csv}")

    # Print summary
    print(f"\n{'='*60}")
    print("Statistical Summary (panel c):")
    print(f"{'='*60}")
    for test_cond in ordered_test_conds:
        result = perm_results[test_cond]
        print(f"\n{test_cond} vs {control_condition}:")
        print(f"  Raw significant bins: {result['n_significant_raw']}/{n_bins}")
        print(f"  FDR significant bins: {result['n_significant']}/{n_bins}")
        if result["p_values_fdr"] is not None and len(result["p_values_fdr"]) > 0:
            min_p_fdr = np.nanmin(result["p_values_fdr"])
            print(f"  Minimum FDR-corrected p-value: {min_p_fdr:.4f}")
    print(f"\nExpected: no significant bins (all FDR-corrected p ≥ 0.08)")

    return perm_results


def main():
    """Main function for Extended Data Figure 7c."""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 7c: Ball distance trajectories by surface treatment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of time bins (default: 12, i.e. 5-min segments for 60-min experiment)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical testing (default: 10000)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    output_dir = figure_output_dir("EDFigure7", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 7c: Ball Distance Trajectories")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {args.n_bins} (5-min segments)")
    print(f"Permutations: {args.n_permutations}")

    # Load data
    data = load_coordinates_data(test_mode=args.test)

    # Generate plots and statistics
    generate_trajectory_plots(
        data,
        control_condition="control",
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=output_dir,
        show_progress=not args.no_progress,
    )

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 7c generated successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
