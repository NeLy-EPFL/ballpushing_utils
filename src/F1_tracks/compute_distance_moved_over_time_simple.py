#!/usr/bin/env python3
"""
Compute distance moved over time for F1 TNT genotypes using dataset directly.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

COORDINATES_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
)
SUMMARY_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather"
)
OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_Over_Time")

SELECTED_GENOTYPES = ["TNTxEmptySplit", "TNTxDDC", "TNTxTH", "TNTxTRH", "TNTxMB247", "TNTxLC10-2", "TNTxLC16-1"]

GENOTYPE_COLORS = {
    "TNTxEmptySplit": "#7f7f7f",
    "TNTx DDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
}

PRETRAINING_STYLES = {
    "Pretrained": {
        "linewidth": 3.0,
        "alpha": 1.0,
        "linestyle": "-",
        "marker": "o",
        "markersize": 7,
        "color_adjust": 1.0,  # Use full genotype color
    },
    "Naive": {
        "linewidth": 2.0,
        "alpha": 0.7,
        "linestyle": "--",
        "marker": "s",
        "markersize": 6,
        "color_adjust": 0.7,  # Use lighter version
    },
}


def load_datasets():
    """Load datasets."""
    print("Loading datasets...")
    df_coords = pd.read_feather(COORDINATES_PATH)
    df_summary = pd.read_feather(SUMMARY_PATH)
    print(f"  F1 coordinates: {df_coords.shape}")
    print(f"  Summary: {df_summary.shape}")
    return df_coords, df_summary


def compute_distance_over_time_for_fly(df_fly, fly_name, ball_type="test"):
    """Compute cumulative distance for a single fly."""
    distance_col = "training_ball_euclidean_distance" if ball_type == "training" else "test_ball_euclidean_distance"

    if "interaction_event" not in df_fly.columns:
        return None

    df_interact = df_fly[df_fly["interaction_event"].notna()].copy()
    if df_interact.empty:
        return None

    df_interact = df_interact.sort_values("time").reset_index(drop=True)
    times = df_interact["time"].values.astype(float)
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
    }


def process_genotype(df_coords, genotype):
    """Process all flies in a genotype."""
    df_gen = df_coords[df_coords["Genotype"] == genotype].copy()
    if df_gen.empty or "Pretraining" not in df_gen.columns:
        return {}

    print(f"  Processing {genotype}: {df_gen['fly'].nunique()} flies")
    results = {"genotype": genotype, "pretraining_data": {}}

    for pretrain_val in sorted(df_gen["Pretraining"].dropna().unique()):
        df_pretrain = df_gen[df_gen["Pretraining"] == pretrain_val].copy()
        value_str = str(pretrain_val).lower()
        pretrain_label = "Pretrained" if value_str in ["y", "yes", "true", "1"] else "Naive"

        pretrain_results = {"pretraining": pretrain_label, "flies": {}}
        flies_loaded = 0

        for fly_name in df_pretrain["fly"].unique():
            df_fly = df_pretrain[df_pretrain["fly"] == fly_name].copy()
            result = compute_distance_over_time_for_fly(df_fly, fly_name, ball_type="test")
            if result is not None:
                flies_loaded += 1
                pretrain_results["flies"][fly_name] = result

        print(f"      {pretrain_label}: {flies_loaded} flies with data")
        if pretrain_results["flies"]:
            results["pretraining_data"][pretrain_label] = pretrain_results

    return results


def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    """Compute bootstrapped confidence intervals."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    return (
        np.mean(data),
        np.percentile(bootstrap_means, lower_percentile),
        np.percentile(bootstrap_means, upper_percentile),
    )


def plot_distance_over_time(all_results, output_dir, global_ylim=None):
    """Create plots of averaged distance over time."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_y_values = []

    for genotype, gen_results in all_results.items():
        if "pretraining_data" not in gen_results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")

        for pretrain_label in sorted(gen_results["pretraining_data"].keys()):
            pretrain_results = gen_results["pretraining_data"][pretrain_label]
            if not pretrain_results.get("flies"):
                continue

            all_time_distance_pairs = []
            for fly_name, fly_result in pretrain_results["flies"].items():
                for t, d in zip(fly_result["times"], fly_result["full_cumulative"]):
                    all_time_distance_pairs.append((float(t), float(d)))
                    all_y_values.append(float(d))

            if not all_time_distance_pairs:
                continue

            print(f"    OK {genotype} {pretrain_label}: {len(all_time_distance_pairs)} data points")

            all_time_distance_pairs.sort(key=lambda x: x[0])
            times_array = np.array([p[0] for p in all_time_distance_pairs])
            distances_array = np.array([p[1] for p in all_time_distance_pairs])

            # Use 5-second bins for smoother curves
            time_bins = np.arange(0, times_array.max() + 5, 5.0)
            bin_indices = np.digitize(times_array, time_bins)

            means, lower_ci, upper_ci, bin_centers = [], [], [], []
            for bin_idx in np.unique(bin_indices):
                if bin_idx == 0 or bin_idx > len(time_bins) - 1:
                    continue
                mask = bin_indices == bin_idx
                bin_distances = distances_array[mask]
                if len(bin_distances) > 2:  # Need at least 3 points for bootstrap
                    mean, lower, upper = bootstrap_confidence_interval(bin_distances, n_bootstrap=500)
                    means.append(mean)
                    lower_ci.append(lower)
                    upper_ci.append(upper)
                    bin_centers.append(time_bins[bin_idx - 1] + 2.5)  # Center of bin

            if means:
                pretrain_style = PRETRAINING_STYLES.get(pretrain_label, {})
                # Plot line
                ax.plot(
                    bin_centers,
                    means,
                    linestyle=pretrain_style.get("linestyle", "-"),
                    label=f"{pretrain_label} (n={len(pretrain_results['flies'])})",
                    linewidth=pretrain_style.get("linewidth", 2.5),
                    color=genotype_color,
                    alpha=pretrain_style.get("alpha", 0.9),
                )
                # Add confidence interval shading
                ax.fill_between(
                    bin_centers,
                    lower_ci,
                    upper_ci,
                    color=genotype_color,
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Distance (pixels)", fontsize=13, fontweight="bold")
        ax.set_title(f"{genotype} - Distance Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best", framealpha=0.9, title="95% Bootstrap CI")
        ax.grid(True, alpha=0.3)

        if global_ylim and not any(np.isnan(global_ylim) | np.isinf(global_ylim)):
            ax.set_ylim(global_ylim)

        plt.tight_layout()
        output_path = output_dir / f"distance_over_time_{genotype}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"    Saved {output_path.name}")
        plt.close()

    if all_y_values:
        all_y_values = np.array(all_y_values)
        all_y_values = all_y_values[~np.isnan(all_y_values)]
        if len(all_y_values) > 0:
            y_max = np.max(all_y_values)
            if not np.isnan(y_max) and not np.isinf(y_max):
                return (0, y_max * 1.1)
    return None


def plot_all_genotypes_combined(all_results, output_dir, global_ylim):
    """Create combined grid plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_genotypes = len(all_results)
    n_cols, n_rows = 4, (n_genotypes + 3) // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    if n_genotypes == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flat

    for idx, (genotype, gen_results) in enumerate(sorted(all_results.items())):
        ax = axes_flat[idx]
        if "pretraining_data" not in gen_results:
            ax.axis("off")
            continue

        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")
        for pretrain_label in sorted(gen_results["pretraining_data"].keys()):
            pretrain_results = gen_results["pretraining_data"][pretrain_label]
            if not pretrain_results.get("flies"):
                continue

            all_time_distance_pairs = []
            for fly_name, fly_result in pretrain_results["flies"].items():
                for t, d in zip(fly_result["times"], fly_result["full_cumulative"]):
                    all_time_distance_pairs.append((float(t), float(d)))

            if not all_time_distance_pairs:
                continue

            all_time_distance_pairs.sort(key=lambda x: x[0])
            times_array = np.array([p[0] for p in all_time_distance_pairs])
            distances_array = np.array([p[1] for p in all_time_distance_pairs])

            # Use 5-second bins for smoother curves
            time_bins = np.arange(0, times_array.max() + 5, 5.0)
            bin_indices = np.digitize(times_array, time_bins)

            means, lower_ci, upper_ci, bin_centers = [], [], [], []
            for bin_idx in np.unique(bin_indices):
                if bin_idx == 0 or bin_idx > len(time_bins) - 1:
                    continue
                mask = bin_indices == bin_idx
                bin_distances = distances_array[mask]
                if len(bin_distances) > 2:
                    mean, lower, upper = bootstrap_confidence_interval(bin_distances, n_bootstrap=500)
                    means.append(mean)
                    lower_ci.append(lower)
                    upper_ci.append(upper)
                    bin_centers.append(time_bins[bin_idx - 1] + 2.5)

            if means:
                pretrain_style = PRETRAINING_STYLES.get(pretrain_label, {})
                # Plot line
                ax.plot(
                    bin_centers,
                    means,
                    linestyle=pretrain_style.get("linestyle", "-"),
                    label=f"{pretrain_label} (n={len(pretrain_results['flies'])})",
                    linewidth=pretrain_style.get("linewidth", 2.0),
                    color=genotype_color,
                    alpha=pretrain_style.get("alpha", 0.9),
                )
                # Add confidence interval shading
                ax.fill_between(
                    bin_centers,
                    lower_ci,
                    upper_ci,
                    color=genotype_color,
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_xlabel("Time (sec)", fontsize=10)
        ax.set_ylabel("Distance (px)", fontsize=10)
        ax.set_title(genotype, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        if global_ylim and not any(np.isnan(global_ylim) | np.isinf(global_ylim)):
            ax.set_ylim(global_ylim)

    for idx in range(n_genotypes, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "distance_over_time_all_genotypes.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path.name}")
    plt.close()


def generate_summary_statistics(all_results, output_dir):
    """Generate summary CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for genotype, gen_results in all_results.items():
        if "pretraining_data" not in gen_results:
            continue
        for pretrain_label, pretrain_results in gen_results["pretraining_data"].items():
            for fly_name, fly_result in pretrain_results["flies"].items():
                rows.append(
                    {
                        "Genotype": genotype,
                        "Pretraining": pretrain_label,
                        "Fly": fly_name,
                        "Total_Distance": fly_result.get("total_distance", 0),
                    }
                )

    if rows:
        df_summary = pd.DataFrame(rows)
        output_path = output_dir / "distance_summary.csv"
        df_summary.to_csv(output_path, index=False)
        print(f"  Saved {output_path.name}")


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("COMPUTING DISTANCE MOVED OVER TIME (Using Dataset Directly)")
    print("=" * 70 + "\n")

    df_coords, df_summary = load_datasets()

    required_cols = ["interaction_event", "test_ball_euclidean_distance"]
    if any(col not in df_coords.columns for col in required_cols):
        print("\nError: Missing required columns")
        return False

    print("\nProcessing genotypes...")
    all_results = {}
    for genotype in SELECTED_GENOTYPES:
        results = process_genotype(df_coords, genotype)
        if results:
            all_results[genotype] = results

    if not all_results:
        print("\nError: No results generated!")
        return False

    print("\nGenerating plots...")
    global_ylim = plot_distance_over_time(all_results, OUTPUT_DIR, global_ylim=None)

    if global_ylim is None:
        global_ylim = (0, 1000000)
        print("  Warning: Using default y-axis range")
    else:
        print(f"  Global y-range: {global_ylim[0]:.0f} to {global_ylim[1]:.0f} pixels")

    print("\nApplying global y-axis limits...")
    plot_distance_over_time(all_results, OUTPUT_DIR, global_ylim=global_ylim)

    print("\nGenerating combined genotype layout...")
    plot_all_genotypes_combined(all_results, OUTPUT_DIR, global_ylim)

    print("\nGenerating summary statistics...")
    generate_summary_statistics(all_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("OK Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
