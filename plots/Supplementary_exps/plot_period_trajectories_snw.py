#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for starved_noWater flies,
comparing AM vs PM period.

This script generates trajectory visualizations with:
- starved_noWater FeedingState only
- Light ON condition only (Dark files excluded by filename + Light column filter)
- Short experiments excluded (< 75 % of max recording duration)
- Period as grouping variable: AM vs PM (PM15 folded into PM)
- Downsampling to 1 datapoint per second
- Bootstrapped 95% confidence intervals
- FDR-corrected permutation test across time bins
- Single comparison: AM vs PM

Usage:
    python plot_period_trajectories_snw.py [--n-bins N] [--n-permutations N]
                                            [--output-dir PATH] [--test]
"""

import argparse
import sys
import time
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXELS_PER_MM = 500 / 30  # 500 px = 30 mm

# Short experiments excluded from analysis (only 1-hour recordings)
SHORT_EXPERIMENT_STEMS = [
    "240718_Afternoon_FeedingState_Videos_Tracked",
    "240718_Afternoon_FeedingState_next_Videos_Tracked",
]

PERIOD_ORDER = ["AM", "PM"]
PERIOD_LABELS = {"AM": "AM", "PM": "PM"}
PERIOD_COLORS = {
    "AM": "#E6AB02",  # golden yellow
    "PM": "#7570B3",  # purple
}
LINE_STYLES = {
    "AM": "solid",
    "PM": "dashed",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coordinates_incrementally(coordinates_dir, test_mode=False):
    """
    Load coordinate feather files:
    - Skips Dark/dark files by filename
    - Skips SHORT_EXPERIMENT_STEMS
    - Filters for Light=on, FeedingState=starved_noWater
    - Normalises Period (AM/PM)
    - Downsamples to 1 Hz
    """
    coordinates_dir = Path(coordinates_dir)
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    all_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not all_files:
        raise FileNotFoundError(f"No coordinate feather files found in {coordinates_dir}")

    # Exclude dark and short-experiment files by filename
    non_dark_files = [f for f in all_files if "dark" not in f.name.lower()]
    excluded_dark = len(all_files) - len(non_dark_files)
    valid_files = [f for f in non_dark_files if not any(s in f.stem for s in SHORT_EXPERIMENT_STEMS)]
    excluded_short = len(non_dark_files) - len(valid_files)

    print(f"\n{'='*60}")
    print("LOADING COORDINATE DATASETS (starved_noWater, AM vs PM)")
    print(f"{'='*60}")
    print(f"Total files:          {len(all_files)}")
    print(f"Dark files excluded:  {excluded_dark}")
    print(f"Short exp excluded:   {excluded_short}")
    print(f"Files to load:        {len(valid_files)}")

    if test_mode:
        valid_files = valid_files[:2]
        print(f"TEST MODE: loading first {len(valid_files)} file(s)")

    all_downsampled = []

    for i, file_path in enumerate(valid_files, 1):
        print(f"\n[{i}/{len(valid_files)}] {file_path.name}")
        try:
            df = pd.read_feather(file_path)
            print(f"  Shape: {df.shape}")

            required_cols = ["time", "distance_ball_0", "fly", "Light", "FeedingState", "Period"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  Skipping – missing columns: {missing}")
                continue

            # Light=on filter
            before = len(df)
            df = df[df["Light"] == "on"].copy()
            print(f"  After Light=on: {len(df)} rows (was {before})")
            if len(df) == 0:
                continue

            # FeedingState filter: starved_noWater only
            df["FeedingState"] = df["FeedingState"].str.strip()
            df = df[df["FeedingState"].isin(["starved_noWater", "starved_nowater"])].copy()
            print(f"  After starved_noWater filter: {len(df)} rows")
            if len(df) == 0:
                print("  Skipping – no starved_noWater data")
                continue

            # Normalise Period
            df["Period"] = df["Period"].astype(str).str.strip()
            df["Period"] = df["Period"].apply(
                lambda p: "AM" if "AM" in p.upper() else ("PM" if "PM" in p.upper() else None)
            )
            df = df[df["Period"].isin(["AM", "PM"])].copy()
            print(f"  Period counts: {df['Period'].value_counts().to_dict()}")
            if len(df) == 0:
                print("  Skipping – no usable Period data")
                continue

            # Downsample to 1 Hz
            df = df.sort_values(["fly", "time"]).copy()
            df["time_rounded"] = df["time"].round(0).astype(int)
            agg_cols = {"distance_ball_0": "mean", "Period": "first"}
            downsampled = (
                df.groupby(["fly", "time_rounded"], as_index=False)
                .agg(agg_cols)
                .rename(columns={"time_rounded": "time"})
            )

            # Make fly IDs unique across files
            downsampled["fly"] = file_path.stem + "::" + downsampled["fly"].astype(str)
            print(f"  Downsampled: {downsampled.shape}, flies: {downsampled['fly'].nunique()}")

            if test_mode:
                keep_flies = []
                for p in downsampled["Period"].unique():
                    flies = downsampled[downsampled["Period"] == p]["fly"].unique()[:10]
                    keep_flies.extend(flies)
                downsampled = downsampled[downsampled["fly"].isin(keep_flies)].copy()
                print(f"  TEST MODE: {downsampled['fly'].nunique()} flies retained")

            all_downsampled.append(downsampled)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_downsampled:
        raise ValueError("No datasets loaded successfully.")

    print(f"\n{'='*60}")
    combined = pd.concat(all_downsampled, ignore_index=True)
    print(f"Combined shape: {combined.shape} | Total flies: {combined['fly'].nunique()}")
    for p, cnt in combined["Period"].value_counts().items():
        n_flies = combined[combined["Period"] == p]["fly"].nunique()
        print(f"  {p}: {n_flies} flies, {cnt} datapoints")

    return combined


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="Period", subject_col="fly", n_bins=12
):
    """Bin time, compute per-fly mean per bin."""
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
# Permutation test
# ---------------------------------------------------------------------------


def compute_permutation_test(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="Period",
    n_permutations=10000,
    alpha=0.05,
    show_progress=True,
):
    """
    Permutation test (mean difference) for each time bin: AM vs PM.
    FDR correction (BH) across bins.
    """
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

    obs_diffs, p_raw = [], []
    iterator = tqdm(time_bins, desc="  AM vs PM") if show_progress else time_bins

    for tb in iterator:
        bin_data = processed_data[processed_data["time_bin"] == tb]
        vals_am = bin_data[bin_data[group_col] == "AM"][metric].values
        vals_pm = bin_data[bin_data[group_col] == "PM"][metric].values

        if len(vals_am) == 0 or len(vals_pm) == 0:
            obs_diffs.append(np.nan)
            p_raw.append(1.0)
            continue

        obs_diff = np.mean(vals_pm) - np.mean(vals_am)
        obs_diffs.append(obs_diff)

        combined = np.concatenate([vals_am, vals_pm])
        n_am = len(vals_am)
        perm_diffs = np.empty(n_permutations)
        for k in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[k] = np.mean(combined[n_am:]) - np.mean(combined[:n_am])

        p_raw.append(float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))))

    p_raw = np.array(p_raw)
    rejected, p_corr, _, _ = multipletests(p_raw, alpha=alpha, method="fdr_bh")

    result = {
        "time_bins": time_bins,
        "observed_diffs": obs_diffs,
        "p_values_raw": p_raw,
        "p_values_corrected": p_corr,
        "significant_timepoints": np.where(rejected)[0].tolist(),
        "n_significant": int(np.sum(rejected)),
        "n_significant_raw": int(np.sum(p_raw < alpha)),
    }
    print(f"    Raw significant bins:  {result['n_significant_raw']}/{n_bins}")
    print(f"    FDR significant bins:  {result['n_significant']}/{n_bins}  (α={alpha})")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_trajectory_plot(
    data,
    perm_result,
    time_col="time",
    value_col="distance_ball_0",
    group_col="Period",
    subject_col="fly",
    n_bins=12,
    output_path=None,
    n_bootstrap=1000,
):
    """Single figure, AM and PM overlaid, with significance annotations."""
    present_conditions = [c for c in PERIOD_ORDER if c in data[group_col].unique()]

    data = data.copy()
    value_mm = f"{value_col}_mm"
    data[value_mm] = data[value_col] / PIXELS_PER_MM

    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    fig, ax = plt.subplots(figsize=(300 / 25.4, 150 / 25.4))

    font_size_ticks = 9
    font_size_labels = 12
    font_size_legend = 9
    font_size_ann = 11

    # Bootstrap CI per condition
    all_means = []
    condition_curves = {}

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
                bs = [np.mean(np.random.choice(fly_vals, size=len(fly_vals), replace=True)) for _ in range(n_bootstrap)]
                ci_lo, ci_hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            else:
                ci_lo = ci_hi = mean_val
            time_stats.append({time_col: t, "mean": mean_val, "ci_lo": ci_lo, "ci_hi": ci_hi})

        df_curve = pd.DataFrame(time_stats)
        condition_curves[cond] = df_curve
        all_means.extend(df_curve["mean"].values)

    y_baseline = min(all_means)

    for cond in present_conditions:
        df_curve = condition_curves[cond]
        n_flies = data[data[group_col] == cond][subject_col].nunique()
        color = PERIOD_COLORS.get(cond, "black")
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
            label=f"{PERIOD_LABELS.get(cond, cond)} (n={n_flies})",
            zorder=10,
        )
        ax.fill_between(df_curve[time_col], shifted_lo, shifted_hi, color=color, alpha=0.2, zorder=5)

    # Vertical bin-edge lines
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, linewidth=0.8, zorder=1)

    # Y limits + annotations
    y_max_shifted = max(all_means) - y_baseline
    y_range = max(y_max_shifted, 1e-6)
    ann_height = 0.07 * y_range
    ax.set_ylim(bottom=-0.02 * y_range, top=y_max_shifted + 1.5 * ann_height)

    ann_color = PERIOD_COLORS["PM"]
    y_ann = y_max_shifted + 0.3 * ann_height
    for bin_idx in perm_result["significant_timepoints"]:
        if bin_idx >= len(bin_edges) - 1:
            continue
        bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        p = perm_result["p_values_corrected"][bin_idx]
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
            color=ann_color,
        )

    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=font_size_labels)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.grid(False)

    # Legend outside
    legend_elements = [
        Patch(
            facecolor=PERIOD_COLORS.get(c, "black"),
            edgecolor="black",
            linewidth=0.5,
            label=f"{PERIOD_LABELS.get(c, c)} (n={data[data[group_col]==c][subject_col].nunique()})",
        )
        for c in present_conditions
    ]
    legend_elements.append(Patch(visible=False, label=""))
    legend_elements.append(Patch(facecolor=PERIOD_COLORS["PM"], edgecolor="none", alpha=0.8, label="* AM vs PM"))
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
    data, n_bins=12, n_permutations=10000, output_dir=None, show_progress=True, n_bootstrap=1000
):
    if output_dir is None:
        output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_period_snw")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AM vs PM TRAJECTORY ANALYSIS  (starved_noWater, Light ON)")
    print(f"{'='*60}")
    print(f"Output:       {output_dir}")
    print(f"Time bins:    {n_bins}")
    print(f"Permutations: {n_permutations}")
    print(f"Bootstrap:    {n_bootstrap} resamples")

    # Convert seconds → minutes
    data = data.copy()
    data["time"] = data["time"] / 60.0
    print(f"Time range: {data['time'].min():.2f} – {data['time'].max():.2f} min")

    # Drop short-recording flies
    time_max = data["time"].max()
    min_required = 0.75 * time_max
    fly_max_time = data.groupby("fly")["time"].max()
    short_flies = fly_max_time[fly_max_time < min_required].index
    if len(short_flies):
        before = data["fly"].nunique()
        data = data[~data["fly"].isin(short_flies)].copy()
        print(f"Dropped {len(short_flies)} short-recording flies. " f"{data['fly'].nunique()}/{before} retained.")
    else:
        print(f"All flies span ≥{min_required:.0f} min. None dropped.")

    print(f"\nBinning into {n_bins} time bins …")
    processed = preprocess_data(data, n_bins=n_bins)
    print(f"Processed shape: {processed.shape}")

    print(f"\nComputing permutation test ({n_permutations:,} permutations, FDR-BH) …")
    perm_result = compute_permutation_test(processed, n_permutations=n_permutations, show_progress=show_progress)

    print("\nGenerating trajectory plot …")
    output_path = output_dir / "trajectory_period_snw_AM_vs_PM.pdf"
    create_trajectory_plot(data, perm_result, n_bins=n_bins, output_path=output_path, n_bootstrap=n_bootstrap)

    # Save stats
    stats_file = output_dir / "trajectory_period_snw_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("AM vs PM TRAJECTORY PERMUTATION TEST (starved_noWater)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time bins:    {n_bins}\n")
        f.write(f"Permutations: {n_permutations}\n")
        f.write(f"FDR method:   fdr_bh\n\n")
        f.write(f"Raw significant bins:  {perm_result['n_significant_raw']}/{n_bins}\n")
        f.write(f"FDR significant bins:  {perm_result['n_significant']}/{n_bins}\n")
        f.write(f"Significant bin indices: {perm_result['significant_timepoints']}\n\n")
        f.write("Per-bin results:\n")
        for i, tb in enumerate(perm_result["time_bins"]):
            sig = "*" if i in perm_result["significant_timepoints"] else ""
            obs = perm_result["observed_diffs"][i]
            pr = perm_result["p_values_raw"][i]
            pc = perm_result["p_values_corrected"][i]
            f.write(f"  Bin {tb:2d}: diff={obs:+.2f}, p_raw={pr:.4f}, p_FDR={pc:.4f} {sig}\n")
    print(f"\nStatistics saved: {stats_file}")

    print(f"\n{'='*60}")
    print(f"Done. Results saved to: {output_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ball trajectory plots: starved_noWater AM vs PM (Light ON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--coordinates-dir",
        type=str,
        default=(
            "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
            "260220_10_summary_control_folders_Data/coordinates"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_period_snw",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test mode: first 2 files, 10 flies per condition")

    args = parser.parse_args()

    t0 = time.time()
    data = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"\nData loaded in {time.time() - t0:.1f}s")

    generate_trajectory_plots(
        data,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
