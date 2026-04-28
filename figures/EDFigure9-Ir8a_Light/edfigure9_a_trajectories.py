#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 9a: Ball distance trajectories for IR8a
flies with and without illumination.

This script generates a trajectory visualization showing:
- Distance the ball was pushed over 2 h of the experiment
- IR8a only: Light ON (orange) vs Light OFF (black)
- Time-series data binned into 5 min segments (24 bins × 5 min = 120 min)
- Permutation tests with FDR correction (α = 0.05)
- Bootstrapped 95% confidence intervals

Expected results (from published figure):
  - Significant difference in bins 0–3 (first 20 min): FDR p < 0.001 / p = 0.015
  - No significant difference in bins 4–11 (20–120 min): all p ≥ 0.23

Usage:
    python edfigure9_a_trajectories.py [--test] [--n-bins N] [--n-permutations N]
"""

import argparse
import time
from pathlib import Path


from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXELS_PER_MM = 500 / 30  # 500 px = 30 mm

COORDINATES_PATH = (
    dataset("TNT_Olfaction_Dark/Datasets/251106_08_summary_TNT_Olfaction_Dark_Data/coordinates/pooled_coordinates.feather")
)

# Light ON = orange, Light OFF = black
LIGHT_COLORS = {"on": "orange", "off": "black"}
LIGHT_LABELS = {"on": "Light ON", "off": "Light OFF"}
ERROR_ALPHAS = {"on": 0.25, "off": 0.25}
ERROR_COLORS = {"on": "orange", "off": "dimgray"}


# ---------------------------------------------------------------------------
# Statistical helpers
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


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrapped CI for the mean difference (group2 − group1)."""
    if len(group1) == 0 or len(group2) == 0:
        return None, None
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs[i] = np.mean(s2) - np.mean(s1)
    alpha = 100 - ci
    return float(np.percentile(diffs, alpha / 2)), float(np.percentile(diffs, 100 - alpha / 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coordinates(coordinates_path, test_mode=False):
    """
    Load and downsample the dark olfaction coordinates dataset.

    Filters to IR8a genotype only and downsample to 1 Hz.
    """
    coordinates_path = Path(coordinates_path)
    if not coordinates_path.exists():
        raise FileNotFoundError(f"Coordinates not found: {coordinates_path}")

    print(f"Loading coordinates from: {coordinates_path}")
    df = read_feather(coordinates_path)
    print(f"Loaded: {df.shape}")

    # Check required columns
    required = ["time", "distance_ball_0", "Light", "Genotype", "fly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter to IR8a only
    df = df[df["Genotype"] == "TNTxIR8a"].copy()
    print(f"After IR8a filter: {df.shape}")

    # Filter to Light on/off only
    df = df[df["Light"].isin(["on", "off"])].copy()
    print(f"After Light on/off filter: {df.shape}")

    # Drop rows with NaN in essential columns
    df = df.dropna(subset=["time", "distance_ball_0", "Light", "fly"]).copy()
    print(f"After NaN drop: {df.shape}")

    if test_mode:
        # Limit to 10 flies per light condition
        limited = []
        for light in df["Light"].unique():
            flies = df[df["Light"] == light]["fly"].unique()[:10]
            limited.extend(flies)
        df = df[df["fly"].isin(limited)].copy()
        print(f"TEST MODE: {df['fly'].nunique()} flies")

    # Downsample to 1 Hz by rounding time to nearest integer second
    df = df.sort_values(["fly", "time"]).copy()
    df["time_rounded"] = df["time"].round(0).astype(int)
    downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg({"distance_ball_0": "mean", "Light": "first"})
    downsampled = downsampled.rename(columns={"time_rounded": "time"})

    print(f"Downsampled shape: {downsampled.shape}")
    for light in sorted(downsampled["Light"].unique()):
        n = downsampled[downsampled["Light"] == light]["fly"].nunique()
        print(f"  Light {light}: {n} flies")

    return downsampled


# ---------------------------------------------------------------------------
# Preprocessing / binning
# ---------------------------------------------------------------------------


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="Light", subject_col="fly", n_bins=24
):
    """Bin time and compute per-fly mean per bin."""
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    data = data.copy()
    data["time_bin"] = pd.cut(data[time_col], bins=bin_edges, labels=range(n_bins), include_lowest=True)
    data["time_bin"] = data["time_bin"].astype(int)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    grouped = data.groupby([group_col, subject_col, "time_bin"])[value_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{value_col}"]
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))

    return grouped, bin_edges


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


def compute_permutation_tests_with_fdr(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="Light",
    control_group="on",
    test_group="off",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Permutation tests per time bin with FDR correction (Benjamini-Hochberg).

    control_group = "on" (light on), test_group = "off" (light off).
    """
    print(f"\nPermutation tests: {test_group} vs {control_group}")
    comparison_data = processed_data[processed_data[group_col].isin([control_group, test_group])]
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

    observed_diffs = []
    p_values_raw = []
    n_control_values = []
    n_test_values = []
    mean_control_values = []
    mean_test_values = []
    ci_lower_values = []
    ci_upper_values = []
    cohens_d_values = []

    rng = np.random.default_rng(42)
    iterator = tqdm(time_bins, desc="  bins") if progress else time_bins

    for time_bin in iterator:
        bin_data = comparison_data[comparison_data["time_bin"] == time_bin]
        ctrl = bin_data[bin_data[group_col] == control_group][metric].values
        test = bin_data[bin_data[group_col] == test_group][metric].values

        n_control_values.append(len(ctrl))
        n_test_values.append(len(test))

        if len(ctrl) == 0 or len(test) == 0:
            observed_diffs.append(np.nan)
            p_values_raw.append(1.0)
            mean_control_values.append(np.nan)
            mean_test_values.append(np.nan)
            ci_lower_values.append(np.nan)
            ci_upper_values.append(np.nan)
            cohens_d_values.append(np.nan)
            continue

        obs_diff = np.mean(test) - np.mean(ctrl)
        observed_diffs.append(obs_diff)
        mean_control_values.append(float(np.mean(ctrl)))
        mean_test_values.append(float(np.mean(test)))

        ci_lower, ci_upper = bootstrap_ci_difference(ctrl, test, n_bootstrap=1000)
        ci_lower_values.append(ci_lower)
        ci_upper_values.append(ci_upper)
        cohens_d_values.append(cohens_d(ctrl, test))

        # Permutation test
        pooled = np.concatenate([ctrl, test])
        n_ctrl = len(ctrl)
        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(pooled)
            perm_diffs[i] = np.mean(shuffled[n_ctrl:]) - np.mean(shuffled[:n_ctrl])
        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
        p_values_raw.append(p_value)

    p_values_raw = np.array(p_values_raw)
    rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

    results = {
        "time_bins": time_bins,
        "observed_diffs": observed_diffs,
        "n_control": n_control_values,
        "n_test": n_test_values,
        "mean_control": mean_control_values,
        "mean_test": mean_test_values,
        "ci_lower": ci_lower_values,
        "ci_upper": ci_upper_values,
        "cohens_d": cohens_d_values,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_values_corrected,
        "significant_timepoints": np.where(rejected)[0],
        "n_significant": int(np.sum(rejected)),
        "n_significant_raw": int(np.sum(p_values_raw < alpha)),
    }

    print(f"  Raw significant bins: {results['n_significant_raw']}/{n_bins}")
    print(f"  FDR significant bins: {results['n_significant']}/{n_bins} (α={alpha})")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_trajectory_plot(
    data,
    control_condition="on",
    test_condition="off",
    time_col="time",
    value_col="distance_ball_0",
    group_col="Light",
    subject_col="fly",
    n_bins=24,
    permutation_results=None,
    bin_edges=None,
    output_path=None,
):
    """
    Create EDFigure 9a: trajectory plot comparing light conditions for IR8a flies.
    """
    subset = data[data[group_col].isin([control_condition, test_condition])].copy()
    if len(subset) == 0:
        print(f"Warning: no data for {test_condition} vs {control_condition}")
        return

    # Convert pixels to mm
    subset[f"{value_col}_mm"] = subset[value_col] / PIXELS_PER_MM
    value_mm = f"{value_col}_mm"

    fig, ax = plt.subplots(figsize=(10, 6))

    font_size_ticks = 10
    font_size_labels = 14
    font_size_legend = 12

    n_ctrl = subset[subset[group_col] == control_condition][subject_col].nunique()
    n_test = subset[subset[group_col] == test_condition][subject_col].nunique()

    labels = {
        control_condition: f"Light ON (n={n_ctrl})",
        test_condition: f"Light OFF (n={n_test})",
    }

    # Compute y-baseline (minimum mean value across both conditions)
    all_means = []
    for cond in [control_condition, test_condition]:
        cond_data = subset[subset[group_col] == cond]
        time_grouped = cond_data.groupby(time_col)[value_mm].mean().reset_index()
        all_means.extend(time_grouped[value_mm].values)
    y_baseline = min(all_means)

    # Plot mean trajectories with parametric 95% CI (1.96 × SEM)
    for cond in [control_condition, test_condition]:
        cond_data = subset[subset[group_col] == cond]
        fly_agg = cond_data.groupby([subject_col, time_col])[value_mm].mean().reset_index()
        time_points = sorted(fly_agg[time_col].unique())

        means, ci_lows, ci_highs = [], [], []
        for t in time_points:
            vals = fly_agg[fly_agg[time_col] == t][value_mm].values
            if len(vals) > 1:
                m = np.mean(vals)
                sem = stats.sem(vals)
                ci_lows.append(m - 1.96 * sem)
                ci_highs.append(m + 1.96 * sem)
            elif len(vals) == 1:
                m = vals[0]
                ci_lows.append(m)
                ci_highs.append(m)
            else:
                m = np.nan
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
            means.append(m)

        means = np.array(means) - y_baseline
        ci_lows = np.array(ci_lows) - y_baseline
        ci_highs = np.array(ci_highs) - y_baseline

        ax.plot(
            time_points,
            means,
            color=LIGHT_COLORS[cond],
            linewidth=2,
            label=labels[cond],
        )
        ax.fill_between(
            time_points,
            ci_lows,
            ci_highs,
            color=ERROR_COLORS[cond],
            alpha=ERROR_ALPHAS[cond],
            edgecolor="none",
        )

    # Draw vertical dotted lines for time bins
    if bin_edges is not None:
        for edge in bin_edges:
            ax.axvline(x=edge, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    # Significance annotations
    y_max_shifted = max(all_means) - y_baseline
    y_range = y_max_shifted if y_max_shifted > 0 else 1.0
    ax.set_ylim(0, y_max_shifted + 0.12 * y_range)
    y_annot = y_max_shifted + 0.05 * y_range

    if permutation_results is not None and bin_edges is not None:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for bin_idx, p_val in enumerate(permutation_results["p_values_corrected"]):
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            else:
                continue
            ax.text(
                bin_centers[bin_idx],
                y_annot,
                marker,
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=font_size_labels)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.legend(fontsize=font_size_legend, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved trajectory plot to: {output_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 9a: IR8a light trajectory plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=24,
        help="Number of time bins (default: 24 = 5-min segments over 2 h)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations (default: 10000)",
    )
    args = parser.parse_args()

    output_dir = figure_output_dir("EDFigure9", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 9a: IR8a Light Trajectories")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading coordinates...")
    t0 = time.time()
    data = load_coordinates(COORDINATES_PATH, test_mode=args.test)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Convert time from seconds to minutes
    data = data.copy()
    data["time"] = data["time"] / 60.0
    print(f"Time range: {data['time'].min():.2f}–{data['time'].max():.2f} min")

    n_bins = args.n_bins

    # Preprocess
    print(f"\nPreprocessing into {n_bins} time bins ({120 / n_bins:.0f}-min each)...")
    processed, bin_edges = preprocess_data(
        data,
        time_col="time",
        value_col="distance_ball_0",
        group_col="Light",
        subject_col="fly",
        n_bins=n_bins,
    )
    print(f"Processed shape: {processed.shape}")

    # Permutation tests with FDR
    print(f"\nComputing permutation tests ({args.n_permutations:,} permutations, FDR correction)...")
    perm_results = compute_permutation_tests_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="Light",
        control_group="on",
        test_group="off",
        n_permutations=args.n_permutations,
        alpha=0.05,
        progress=True,
    )

    # Save statistics CSV
    stats_rows = []
    for bin_idx in perm_results["time_bins"]:
        bin_data = processed[processed["time_bin"] == bin_idx]
        stats_rows.append(
            {
                "time_bin": bin_idx + 1,
                "bin_start_min": float(bin_data["bin_start"].iloc[0]) if len(bin_data) > 0 else np.nan,
                "bin_end_min": float(bin_data["bin_end"].iloc[0]) if len(bin_data) > 0 else np.nan,
                "n_light_on": perm_results["n_control"][bin_idx],
                "n_light_off": perm_results["n_test"][bin_idx],
                "mean_light_on_mm": (
                    perm_results["mean_control"][bin_idx] / PIXELS_PER_MM
                    if perm_results["mean_control"][bin_idx] is not None
                    and not np.isnan(perm_results["mean_control"][bin_idx])
                    else np.nan
                ),
                "mean_light_off_mm": (
                    perm_results["mean_test"][bin_idx] / PIXELS_PER_MM
                    if perm_results["mean_test"][bin_idx] is not None
                    and not np.isnan(perm_results["mean_test"][bin_idx])
                    else np.nan
                ),
                "difference_mm": (
                    perm_results["observed_diffs"][bin_idx] / PIXELS_PER_MM
                    if perm_results["observed_diffs"][bin_idx] is not None
                    and not np.isnan(perm_results["observed_diffs"][bin_idx])
                    else np.nan
                ),
                "cohens_d": perm_results["cohens_d"][bin_idx],
                "p_value_raw": perm_results["p_values_raw"][bin_idx],
                "p_value_fdr": perm_results["p_values_corrected"][bin_idx],
                "significant_fdr": bool(perm_results["p_values_corrected"][bin_idx] < 0.05),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_file = output_dir / "edfigure9a_trajectory_statistics.csv"
    stats_df.to_csv(stats_file, index=False, float_format="%.6f")
    print(f"\n✅ Statistics saved to: {stats_file}")

    # Print significant bins
    sig = stats_df[stats_df["significant_fdr"]]
    if len(sig) > 0:
        print(f"\nSignificant time bins (FDR p < 0.05):")
        for _, row in sig.iterrows():
            print(
                f"  Bin {int(row['time_bin'])}: "
                f"{row['bin_start_min']:.1f}–{row['bin_end_min']:.1f} min, "
                f"p_fdr={row['p_value_fdr']:.4f}"
            )
    else:
        print("\nNo significant time bins found")

    # Generate plot
    print("\nGenerating trajectory plot...")
    plot_file = output_dir / "edfigure9a_ir8a_light_trajectories.pdf"
    create_trajectory_plot(
        data,
        control_condition="on",
        test_condition="off",
        time_col="time",
        value_col="distance_ball_0",
        group_col="Light",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=perm_results,
        bin_edges=bin_edges,
        output_path=plot_file,
    )

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 9a generated successfully!")
    print(f"{'='*60}")
    print(f"  Plot:  {plot_file}")
    print(f"  Stats: {stats_file}")


if __name__ == "__main__":
    main()
