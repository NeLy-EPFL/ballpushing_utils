#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for FeedingState experiments.

This script generates trajectory visualizations with:
- Light ON condition only (Dark files excluded by filename; Light column also filtered)
- FeedingState as grouping variable: fed (control), starved, starved_noWater
- Downsampling to 1 datapoint per second (from ~30 fps)
- Bootstrapped 95% confidence intervals
- FDR-corrected permutation tests across 12 time bins
- Significance annotations (*, **, ***)
- All FeedingState conditions overlaid on a single figure

Usage:
    python plot_feedingstate_trajectories.py [--n-bins N] [--n-permutations N]
                                              [--output-dir PATH] [--test]

Arguments:
    --n-bins:          Number of time bins for analysis (default: 12)
    --n-permutations:  Number of permutations for statistical testing (default: 10000)
    --output-dir:      Directory to save plots
    --test:            Test mode - only first 2 files, 10 flies per condition
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from ballpushing_utils import read_feather

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXELS_PER_MM = 500 / 30  # 500 px = 30 mm in the setup

# FeedingState canonical names and ordering
FEEDING_ORDER = ["fed", "starved", "starved_noWater"]
FEEDING_LABELS = {
    "fed": "Fed",
    "starved": "Starved",
    "starved_noWater": "Starved (no water)",
}
# Colour palette matching run_permutation_feeding_state.py (seaborn Set2)
FEEDING_COLORS = {
    "fed": "#66C2A5",  # teal
    "starved": "#FC8D62",  # orange
    "starved_noWater": "#8DA0CB",  # blue
}
LINE_STYLES = {
    "fed": "solid",
    "starved": "dashed",
    "starved_noWater": "dotted",
}
# Distinct color per pairwise comparison (annotation rows).
# The third pair reuses neither condition colour to avoid confusion.
PAIR_COLORS = {
    ("fed", "starved"): "#FC8D62",  # starved orange
    ("fed", "starved_noWater"): "#8DA0CB",  # starved_noWater blue
    ("starved", "starved_noWater"): "#984EA3",  # distinct purple
}
PAIR_LABELS = {
    ("fed", "starved"): "Fed vs Starved",
    ("fed", "starved_noWater"): "Fed vs Starved (no water)",
    ("starved", "starved_noWater"): "Starved vs Starved (no water)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coordinates_incrementally(coordinates_dir, test_mode=False):
    """
    Load coordinate feather files, skipping any Dark/dark experiments.
    Filters for Light=on, normalises FeedingState, and downsamples to 1 Hz.

    Parameters
    ----------
    coordinates_dir : str or Path
    test_mode : bool
        If True, only load first 2 files and cap at 10 flies per FeedingState.

    Returns
    -------
    pd.DataFrame
        Combined, downsampled dataset with columns: fly, time, distance_ball_0,
        FeedingState.
    """
    coordinates_dir = Path(coordinates_dir)
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    all_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not all_files:
        raise FileNotFoundError(f"No coordinate feather files found in {coordinates_dir}")

    # Exclude files containing "Dark" or "dark" in the filename
    non_dark_files = [f for f in all_files if "dark" not in f.name.lower()]
    excluded = len(all_files) - len(non_dark_files)

    print(f"\n{'='*60}")
    print("LOADING COORDINATE DATASETS")
    print(f"{'='*60}")
    print(f"Total files found:   {len(all_files)}")
    print(f"Dark files excluded: {excluded}")
    print(f"Files to load:       {len(non_dark_files)}")

    if test_mode:
        non_dark_files = non_dark_files[:2]
        print(f"TEST MODE: loading only first {len(non_dark_files)} file(s)")

    all_downsampled = []

    for i, file_path in enumerate(non_dark_files, 1):
        print(f"\n[{i}/{len(non_dark_files)}] {file_path.name}")
        try:
            df = read_feather(file_path)
            print(f"  Shape: {df.shape}")

            required_cols = ["time", "distance_ball_0", "fly"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  Skipping – missing columns: {missing}")
                continue

            if "Light" not in df.columns:
                print("  Skipping – no 'Light' column")
                continue

            # Filter for Light=on only
            before = len(df)
            df = df[df["Light"] == "on"].copy()
            print(f"  After Light=on filter: {len(df)} rows (was {before})")

            if len(df) == 0:
                print("  Skipping – no Light=on data")
                continue

            # Normalise FeedingState
            if "FeedingState" not in df.columns:
                print("  Skipping – no 'FeedingState' column")
                continue

            df["FeedingState"] = df["FeedingState"].str.strip()
            # Unify capitalisation: "Fed" → "fed"
            rename_map = {
                "Fed": "fed",
                "fed": "fed",
                "starved": "starved",
                "starved_noWater": "starved_noWater",
            }
            df["FeedingState"] = df["FeedingState"].map(rename_map).fillna(df["FeedingState"])

            # Drop rows with unrecognised FeedingState
            before = len(df)
            df = df[df["FeedingState"].isin(FEEDING_ORDER)].copy()
            if len(df) < before:
                print(f"  Dropped {before - len(df)} rows with unrecognised FeedingState")

            if len(df) == 0:
                print("  Skipping – no usable FeedingState data")
                continue

            print(f"  FeedingState counts: {df['FeedingState'].value_counts().to_dict()}")
            print(f"  Unique flies: {df['fly'].nunique()}")

            # Downsample to 1 datapoint per second
            print(f"  Downsampling {len(df)} rows → ~1 Hz …")
            df = df.sort_values(["fly", "time"]).copy()
            df["time_rounded"] = df["time"].round(0).astype(int)

            agg_cols = {"distance_ball_0": "mean", "FeedingState": "first"}
            if "Period" in df.columns:
                # Normalise Period before aggregating
                df["Period"] = df["Period"].astype(str).str.strip()
                df["Period"] = df["Period"].apply(
                    lambda p: "AM" if "AM" in p.upper() else ("PM" if "PM" in p.upper() else None)
                )
                agg_cols["Period"] = "first"
            downsampled = (
                df.groupby(["fly", "time_rounded"], as_index=False)
                .agg(agg_cols)
                .rename(columns={"time_rounded": "time"})
            )
            # Make fly IDs unique across files to prevent collision when
            # multiple files share the same fly identifiers or overlapping
            # time ranges (e.g. recordings split into 1-hour sessions).
            downsampled["fly"] = file_path.stem + "::" + downsampled["fly"].astype(str)
            print(f"  Downsampled shape: {downsampled.shape}")

            # Test mode: limit to 10 flies per FeedingState
            if test_mode:
                keep_flies = []
                for fs in downsampled["FeedingState"].unique():
                    flies = downsampled[downsampled["FeedingState"] == fs]["fly"].unique()[:10]
                    keep_flies.extend(flies)
                downsampled = downsampled[downsampled["fly"].isin(keep_flies)].copy()
                print(f"  TEST MODE: limited to {downsampled['fly'].nunique()} flies")

            all_downsampled.append(downsampled)

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            continue

    if not all_downsampled:
        raise ValueError("No datasets loaded successfully.")

    print(f"\n{'='*60}")
    print("COMBINING DATASETS")
    print(f"{'='*60}")
    combined = pd.concat(all_downsampled, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    print(f"Total flies: {combined['fly'].nunique()}")
    for fs, cnt in combined["FeedingState"].value_counts().items():
        n_flies = combined[combined["FeedingState"] == fs]["fly"].nunique()
        print(f"  {fs}: {n_flies} flies, {cnt} datapoints")

    return combined


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="FeedingState", subject_col="fly", n_bins=12
):
    """
    Bin time, compute per-fly mean per bin.

    Returns
    -------
    pd.DataFrame  with columns: FeedingState, fly, time_bin, avg_distance_ball_0,
                                bin_center, bin_start, bin_end
    """
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    data = data.copy()
    data["time_bin"] = pd.cut(data[time_col], bins=bin_edges, labels=range(n_bins), include_lowest=True).astype(int)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    grouped = data.groupby([group_col, subject_col, "time_bin"])[value_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{value_col}"]
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))

    return grouped


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group

    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Calculate bootstrapped confidence interval for difference between groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group
    n_bootstrap : int
        Number of bootstrap resamples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI for group2 - group1
    """
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)

    return lower, upper


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


def compute_permutation_tests(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="FeedingState",
    control_group="fed",
    n_permutations=10000,
    alpha=0.05,
    show_progress=True,
):
    """
    Permutation test (mean difference) for each time bin, for all pairwise
    condition combinations. FDR correction (BH) across bins for each pair.
    Also computes Cohen's d, bootstrapped CIs, and sample sizes.

    Returns
    -------
    dict  keyed by (cond_a, cond_b) → detailed statistics including:
          - time_bins, observed_diffs, p_values_raw, p_values_corrected
          - cohens_d, ci_lower, ci_upper, pct_change, pct_ci_lower, pct_ci_upper
          - n_a_per_bin, n_b_per_bin, mean_a_per_bin, mean_b_per_bin
          - significant_timepoints, n_significant
    """
    groups = sorted(processed_data[group_col].unique())
    present = [g for g in FEEDING_ORDER if g in groups]
    all_pairs = list(combinations(present, 2))

    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)
    results = {}

    for cond_a, cond_b in all_pairs:
        print(f"\n  Permutation test: {cond_b} vs {cond_a} …")
        comp_data = processed_data[processed_data[group_col].isin([cond_a, cond_b])]

        obs_diffs = []
        p_raw = []
        cohens_d_vals = []
        ci_lower_vals = []
        ci_upper_vals = []
        pct_change_vals = []
        pct_ci_lower_vals = []
        pct_ci_upper_vals = []
        n_a_per_bin = []
        n_b_per_bin = []
        mean_a_per_bin = []
        mean_b_per_bin = []

        iterator = tqdm(time_bins, desc=f"  {cond_a} vs {cond_b}") if show_progress else time_bins

        for tb in iterator:
            bin_data = comp_data[comp_data["time_bin"] == tb]
            vals_a = bin_data[bin_data[group_col] == cond_a][metric].values
            vals_b = bin_data[bin_data[group_col] == cond_b][metric].values

            if len(vals_a) == 0 or len(vals_b) == 0:
                obs_diffs.append(np.nan)
                p_raw.append(1.0)
                cohens_d_vals.append(np.nan)
                ci_lower_vals.append(np.nan)
                ci_upper_vals.append(np.nan)
                pct_change_vals.append(np.nan)
                pct_ci_lower_vals.append(np.nan)
                pct_ci_upper_vals.append(np.nan)
                n_a_per_bin.append(0)
                n_b_per_bin.append(0)
                mean_a_per_bin.append(np.nan)
                mean_b_per_bin.append(np.nan)
                continue

            # Observed difference
            mean_a = np.mean(vals_a)
            mean_b = np.mean(vals_b)
            obs_diff = mean_b - mean_a
            obs_diffs.append(obs_diff)

            # Cohen's d effect size
            d = cohens_d(vals_a, vals_b)
            cohens_d_vals.append(d)

            # Bootstrapped CI for difference
            ci_lower, ci_upper = bootstrap_ci_difference(vals_a, vals_b, n_bootstrap=10000, ci=95, random_state=42)
            ci_lower_vals.append(ci_lower)
            ci_upper_vals.append(ci_upper)

            # Percentage change relative to mean_a
            if mean_a != 0:
                pct_change = (obs_diff / mean_a) * 100
                pct_ci_lower = (ci_lower / mean_a) * 100
                pct_ci_upper = (ci_upper / mean_a) * 100
            else:
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan
            pct_change_vals.append(pct_change)
            pct_ci_lower_vals.append(pct_ci_lower)
            pct_ci_upper_vals.append(pct_ci_upper)

            # Sample sizes and means
            n_a_per_bin.append(len(vals_a))
            n_b_per_bin.append(len(vals_b))
            mean_a_per_bin.append(mean_a)
            mean_b_per_bin.append(mean_b)

            # Permutation test
            combined = np.concatenate([vals_a, vals_b])
            n_a = len(vals_a)
            perm_diffs = np.empty(n_permutations)
            for k in range(n_permutations):
                np.random.shuffle(combined)
                perm_diffs[k] = np.mean(combined[n_a:]) - np.mean(combined[:n_a])

            p_raw.append(float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))))

        p_raw = np.array(p_raw)
        rejected, p_corr, _, _ = multipletests(p_raw, alpha=alpha, method="fdr_bh")

        results[(cond_a, cond_b)] = {
            "cond_a": cond_a,
            "cond_b": cond_b,
            "time_bins": time_bins,
            "observed_diffs": obs_diffs,
            "p_values_raw": p_raw,
            "p_values_corrected": p_corr,
            "cohens_d": cohens_d_vals,
            "ci_lower": ci_lower_vals,
            "ci_upper": ci_upper_vals,
            "pct_change": pct_change_vals,
            "pct_ci_lower": pct_ci_lower_vals,
            "pct_ci_upper": pct_ci_upper_vals,
            "n_a_per_bin": n_a_per_bin,
            "n_b_per_bin": n_b_per_bin,
            "mean_a_per_bin": mean_a_per_bin,
            "mean_b_per_bin": mean_b_per_bin,
            "significant_timepoints": np.where(rejected)[0].tolist(),
            "n_significant": int(np.sum(rejected)),
            "n_significant_raw": int(np.sum(p_raw < alpha)),
        }

        print(f"    Raw significant bins:  {results[(cond_a, cond_b)]['n_significant_raw']}/{n_bins}")
        print(f"    FDR significant bins:  {results[(cond_a, cond_b)]['n_significant']}/{n_bins}  (α={alpha})")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_trajectory_plot(
    data,
    permutation_results,
    control_condition="fed",
    time_col="time",
    value_col="distance_ball_0",
    group_col="FeedingState",
    subject_col="fly",
    n_bins=12,
    output_path=None,
    n_bootstrap=1000,
):
    """
    Single figure with all FeedingState conditions overlaid.
    Shows mean ± bootstrapped 95% CI per condition.
    Significance annotations per test condition vs control, stacked vertically.

    Parameters
    ----------
    data : pd.DataFrame
        Full (1 Hz) trajectory data with FeedingState column.
    permutation_results : dict
        Output of compute_permutation_tests().
    """
    present_conditions = [c for c in FEEDING_ORDER if c in data[group_col].unique()]
    all_pairs = list(combinations(present_conditions, 2))

    # Convert distance to mm
    data = data.copy()
    value_mm = f"{value_col}_mm"
    data[value_mm] = data[value_col] / PIXELS_PER_MM

    # Time already in minutes (converted upstream)
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    # Figure
    fig, ax = plt.subplots(figsize=(300 / 25.4, 150 / 25.4))  # 300×150 mm

    font_size_ticks = 9
    font_size_labels = 12
    font_size_legend = 9
    font_size_ann = 11

    # --- Compute global y-baseline (min of all condition means) ---
    # We'll shift all curves so that the minimum mean == 0
    all_condition_means = []
    condition_curves = {}  # cache computed curves

    for cond in present_conditions:
        cond_data = data[data[group_col] == cond]
        fly_agg = cond_data.groupby([subject_col, time_col])[value_mm].mean().reset_index()

        time_stats = []
        for t in sorted(fly_agg[time_col].unique()):
            fly_vals = fly_agg[fly_agg[time_col] == t][value_mm].values
            if len(fly_vals) == 0:
                continue
            mean_val = np.mean(fly_vals)
            if len(fly_vals) > 1:
                bs_means = [
                    np.mean(np.random.choice(fly_vals, size=len(fly_vals), replace=True)) for _ in range(n_bootstrap)
                ]
                ci_lo = np.percentile(bs_means, 2.5)
                ci_hi = np.percentile(bs_means, 97.5)
            else:
                ci_lo = ci_hi = mean_val

            time_stats.append({time_col: t, "mean": mean_val, "ci_lo": ci_lo, "ci_hi": ci_hi})

        df_curve = pd.DataFrame(time_stats)
        condition_curves[cond] = df_curve
        all_condition_means.extend(df_curve["mean"].values)

    y_baseline = min(all_condition_means)

    # --- Plot each condition ---
    for cond in present_conditions:
        df_curve = condition_curves[cond]
        n_flies = data[data[group_col] == cond][subject_col].nunique()
        label = f"{FEEDING_LABELS.get(cond, cond)} (n={n_flies})"
        color = FEEDING_COLORS.get(cond, "black")
        ls = LINE_STYLES.get(cond, "solid")

        shifted_mean = df_curve["mean"] - y_baseline
        shifted_lo = df_curve["ci_lo"] - y_baseline
        shifted_hi = df_curve["ci_hi"] - y_baseline

        ax.plot(
            df_curve[time_col],
            shifted_mean,
            color=color,
            linestyle=ls,
            linewidth=2.0,
            label=label,
            zorder=10,
        )
        ax.fill_between(
            df_curve[time_col],
            shifted_lo,
            shifted_hi,
            color=color,
            alpha=0.2,
            zorder=5,
        )

    # --- Vertical time-bin lines ---
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, linewidth=0.8, zorder=1)

    # --- Y axis limits with annotation headroom ---
    y_max_shifted = max(all_condition_means) - y_baseline
    y_range = max(y_max_shifted, 1e-6)
    # One annotation row per pair, stacked upward
    ann_height_per_row = 0.07 * y_range
    total_ann_space = ann_height_per_row * (len(all_pairs) + 0.5)
    ax.set_ylim(bottom=-0.02 * y_range, top=y_max_shifted + total_ann_space)

    # --- Significance annotations: one row per pair ---
    for row_idx, (cond_a, cond_b) in enumerate(all_pairs):
        key = (cond_a, cond_b)
        if key not in permutation_results:
            continue
        res = permutation_results[key]
        # Use the pair-specific colour (3 unique colours, one per comparison)
        color = PAIR_COLORS.get((cond_a, cond_b), FEEDING_COLORS.get(cond_b, "gray"))

        y_ann = y_max_shifted + (row_idx + 0.3) * ann_height_per_row

        for bin_idx in res["significant_timepoints"]:
            if bin_idx >= len(bin_edges) - 1:
                continue
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            p = res["p_values_corrected"][bin_idx]
            if p < 0.001:
                marker = "***"
            elif p < 0.01:
                marker = "**"
            elif p < 0.05:
                marker = "*"
            else:
                continue

            ax.text(
                bin_center,
                y_ann,
                marker,
                ha="center",
                va="center",
                fontsize=font_size_ann,
                fontweight="bold",
                color=color,
            )

    # --- Formatting ---
    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=font_size_labels)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.grid(False)

    # Legend: condition swatches + pair significance colour key
    legend_elements = [
        Patch(
            facecolor=FEEDING_COLORS.get(c, "black"),
            edgecolor="black",
            linewidth=0.5,
            label=f"{FEEDING_LABELS.get(c, c)} (n={data[data[group_col]==c][subject_col].nunique()})",
        )
        for c in present_conditions
    ]
    # Blank separator
    legend_elements.append(Patch(visible=False, label=""))
    # One swatch per pairwise comparison with its annotation colour
    for pair in all_pairs:
        color = PAIR_COLORS.get(pair, "gray")
        label = "* " + PAIR_LABELS.get(pair, f"{pair[0]} vs {pair[1]}")
        legend_elements.append(Patch(facecolor=color, edgecolor="none", alpha=0.8, label=label))
    ax.legend(
        handles=legend_elements,
        fontsize=font_size_legend,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.8,
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_trajectory_plots(
    data,
    control_condition="fed",
    n_bins=12,
    n_permutations=10000,
    output_dir=None,
    show_progress=True,
    n_bootstrap=1000,
    period_mode="pooled",
):
    """Run preprocessing, permutation tests, and plotting.

    Parameters
    ----------
    period_mode : {"pooled", "AM", "PM", "both"}
        'pooled'  – use all periods together (default, original behaviour).
        'AM'/'PM' – restrict to that period, save in a *_AM / *_PM subdirectory.
        'both'    – run AM and PM separately into their own subdirectories.
    """
    if output_dir is None:
        base_output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_feeding_state")
    else:
        base_output_dir = Path(output_dir)

    # Build run configs based on period_mode
    run_configs = []  # list of (output_dir, data_subset)
    if period_mode in ("pooled", "both"):
        run_configs.append((base_output_dir, data))
    if period_mode in ("AM", "PM", "both") and "Period" in data.columns:
        periods = ["AM", "PM"] if period_mode == "both" else [period_mode]
        for period in periods:
            period_data = data[data["Period"] == period].copy()
            if len(period_data) == 0:
                print(f"  Period {period}: no data, skipping")
                continue
            period_dir = base_output_dir.parent / (base_output_dir.name + f"_{period}")
            run_configs.append((period_dir, period_data))
            print(f"  Period {period}: {period_data['fly'].nunique()} flies")
    elif period_mode in ("AM", "PM", "both") and "Period" not in data.columns:
        print("  Warning: 'Period' column not found in data – falling back to pooled.")
        if not run_configs:
            run_configs.append((base_output_dir, data))

    if not run_configs:
        print(f"No run configs for period_mode={period_mode!r}. Exiting.")
        return

    for out_dir, subset in run_configs:
        _run_single_trajectory_analysis(
            data=subset,
            control_condition=control_condition,
            n_bins=n_bins,
            n_permutations=n_permutations,
            output_dir=out_dir,
            show_progress=show_progress,
            n_bootstrap=n_bootstrap,
        )


def _run_single_trajectory_analysis(
    data,
    control_condition="fed",
    n_bins=12,
    n_permutations=10000,
    output_dir=None,
    show_progress=True,
    n_bootstrap=1000,
):
    """Internal: run the full trajectory pipeline for one data subset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("FEEDINGSTATE TRAJECTORY ANALYSIS  (Light ON only)")
    print(f"{'='*60}")
    print(f"Output:        {output_dir}")
    print(f"Time bins:     {n_bins}")
    print(f"Permutations:  {n_permutations}")
    print(f"Bootstrap:     {n_bootstrap} resamples")
    print(f"Control:       {control_condition}")

    # Convert time from seconds to minutes
    print("\nConverting time to minutes …")
    data = data.copy()
    data["time"] = data["time"] / 60.0
    print(f"Time range: {data['time'].min():.2f} – {data['time'].max():.2f} min")

    # Drop flies whose recording is shorter than 75 % of the full duration.
    # Short recordings (e.g. 1-hour sessions) cause an apparent drop at the
    # point where those flies disappear from the average.
    time_max = data["time"].max()
    min_required = 0.75 * time_max
    fly_max_time = data.groupby("fly")["time"].max()
    short_flies = fly_max_time[fly_max_time < min_required].index
    if len(short_flies):
        before_flies = data["fly"].nunique()
        data = data[~data["fly"].isin(short_flies)].copy()
        print(
            f"Dropped {len(short_flies)} flies with recordings shorter than "
            f"{min_required:.0f} min (<75 % of {time_max:.0f} min). "
            f"{data['fly'].nunique()}/{before_flies} flies retained."
        )
    else:
        print(f"All flies have recordings spanning ≥{min_required:.0f} min. None dropped.")

    print(f"\nBinning into {n_bins} time bins …")
    processed = preprocess_data(
        data,
        time_col="time",
        value_col="distance_ball_0",
        group_col="FeedingState",
        subject_col="fly",
        n_bins=n_bins,
    )
    print(f"Processed shape: {processed.shape}")

    # Permutation tests
    print(f"\nComputing permutation tests ({n_permutations:,} permutations, FDR-BH) …")
    perm_results = compute_permutation_tests(
        processed,
        metric="avg_distance_ball_0",
        group_col="FeedingState",
        control_group=control_condition,
        n_permutations=n_permutations,
        alpha=0.05,
        show_progress=show_progress,
    )

    # Main trajectory plot
    print("\nGenerating trajectory plot …")
    output_path = output_dir / "trajectory_feedingstate_all.pdf"
    create_trajectory_plot(
        data,
        permutation_results=perm_results,
        control_condition=control_condition,
        time_col="time",
        value_col="distance_ball_0",
        group_col="FeedingState",
        subject_col="fly",
        n_bins=n_bins,
        output_path=output_path,
        n_bootstrap=n_bootstrap,
    )

    # Save statistics text file
    _save_stats(perm_results, processed, n_bins, n_permutations, control_condition, output_dir)

    print(f"\n{'='*60}")
    print("Done. Results saved to:")
    print(f"  {output_dir}")
    print(f"{'='*60}")


def _save_stats(perm_results, processed_data, n_bins, n_permutations, control_condition, output_dir):
    """Save detailed statistical results to CSV files (Nature Neuroscience compatible format).

    Creates one CSV per pairwise comparison with comprehensive statistics.
    """
    output_dir = Path(output_dir)

    # Also save a summary text file with overall results
    stats_file_txt = output_dir / "trajectory_feedingstate_permutation_statistics.txt"
    with open(stats_file_txt, "w") as f:
        f.write("FEEDINGSTATE TRAJECTORY PERMUTATION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Filter:           Light=on only\n")
        f.write(f"Time bins:        {n_bins}\n")
        f.write(f"Permutations:     {n_permutations:,}\n")
        f.write(f"FDR method:       fdr_bh (Benjamini-Hochberg)\n")
        f.write(f"Significance:     α = 0.05\n")
        f.write(f"Bootstrap:        10,000 resamples (95% CI)\n\n")

        for (cond_a, cond_b), res in perm_results.items():
            f.write(f"\n{FEEDING_LABELS.get(cond_b, cond_b)} vs {FEEDING_LABELS.get(cond_a, cond_a)}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Raw significant bins:  {res['n_significant_raw']}/{n_bins}\n")
            f.write(f"FDR significant bins:  {res['n_significant']}/{n_bins}\n\n")

    # Save detailed CSV for each pairwise comparison
    for (cond_a, cond_b), res in perm_results.items():
        csv_filename = f"trajectory_stats_{cond_a}_vs_{cond_b}.csv"
        csv_path = output_dir / csv_filename

        # Build dataframe with all statistics
        df_stats = pd.DataFrame(
            {
                "Time_Bin": res["time_bins"],
                "Mean_Control_mm": res["mean_a_per_bin"],
                "Mean_Test_mm": res["mean_b_per_bin"],
                "N_Control": res["n_a_per_bin"],
                "N_Test": res["n_b_per_bin"],
                "Difference_mm": res["observed_diffs"],
                "CI_Lower_mm": res["ci_lower"],
                "CI_Upper_mm": res["ci_upper"],
                "Pct_Change": res["pct_change"],
                "Pct_CI_Lower": res["pct_ci_lower"],
                "Pct_CI_Upper": res["pct_ci_upper"],
                "Cohens_d": res["cohens_d"],
                "P_Value_Raw": res["p_values_raw"],
                "P_Value_FDR": res["p_values_corrected"],
                "Significant_FDR": [
                    "*" if res["p_values_corrected"][i] < 0.05 else "ns" for i in range(len(res["time_bins"]))
                ],
            }
        )

        df_stats.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"✅ Saved: {csv_path.name}")

    print(f"\n✅ Summary statistics: {stats_file_txt.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ball trajectory plots grouped by FeedingState (Light ON only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_feedingstate_trajectories.py
  python plot_feedingstate_trajectories.py --n-bins 12 --n-permutations 5000
  python plot_feedingstate_trajectories.py --test
        """,
    )
    parser.add_argument("--n-bins", type=int, default=12, help="Number of time bins (default: 12)")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CI (default: 1000)")
    parser.add_argument(
        "--coordinates-dir",
        type=str,
        default=(
            "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
            "260220_10_summary_control_folders_Data/coordinates"
        ),
        help="Directory containing coordinate feather files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/" "trajectories_feeding_state"),
        help="Directory to save plots",
    )
    parser.add_argument("--control", type=str, default="fed", help="Control FeedingState (default: fed)")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--test", action="store_true", help="Test mode: first 2 files, 10 flies per condition")
    parser.add_argument(
        "--period",
        choices=["pooled", "AM", "PM", "both"],
        default="pooled",
        help=(
            "Which data to analyse: "
            "'pooled' (default) = all periods together; "
            "'AM' / 'PM' = only that period; "
            "'both' = AM and PM in separate directories."
        ),
    )

    args = parser.parse_args()

    t0 = time.time()
    data = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"\nData loaded in {time.time() - t0:.1f}s")

    generate_trajectory_plots(
        data,
        control_condition=args.control,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        n_bootstrap=args.n_bootstrap,
        period_mode=args.period,
    )


if __name__ == "__main__":
    main()
