#!/usr/bin/env python3
"""
Trajectory plots for Magnet (y/n) per Genotype.

Loads coordinates from:
  /mnt/upramdya_data/MD/Magneblock_TNT/Datasets/251205_13_summary_magnetblock_TNT_DDC_Data/coordinates/pooled_coordinates.feather

Creates small-multiple line plots (one panel per genotype) showing mean ± CI
for Magnet vs Control, plus per-bin permutation test stars and p-values.

Usage:
  python plot_magnetblock_trajectories_genotype.py [--n-bins N] [--n-permutations N] [--output-dir PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def load_coordinates_dataset():
    dataset_path = (
        "/mnt/upramdya_data/MD/Magneblock_TNT/Datasets/"
        "251205_13_summary_magnetblock_TNT_DDC_Data/coordinates/pooled_coordinates.feather"
    )
    print(f"Loading coordinates dataset from: {dataset_path}")
    df = pd.read_feather(dataset_path)

    required = ["time", "distance_ball_0", "Magnet", "Genotype", "fly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available: {list(df.columns)}")

    # Normalize Magnet and Genotype
    def _norm_magnet(v):
        s = str(v).strip().lower()
        if s in {"n", "no", "false", "0"}:
            return "n"
        if s in {"y", "yes", "true", "1"}:
            return "y"
        return s

    def _norm_genotype(s):
        return " ".join(str(s).strip().split())

    df = df.copy()
    df["Magnet"] = df["Magnet"].apply(_norm_magnet)
    df["Genotype"] = df["Genotype"].apply(_norm_genotype)

    print(f"Magnet groups: {sorted(df['Magnet'].dropna().unique())}")
    print(f"Genotypes: {sorted(df['Genotype'].dropna().unique())}")
    return df


def preprocess_data(
    data,
    time_col="time",
    value_col="distance_ball_0",
    magnet_col="Magnet",
    genotype_col="Genotype",
    subject_col="fly",
    n_bins=12,
):
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    data = data.copy()
    data["time_bin"] = pd.cut(data[time_col], bins=bin_edges, labels=range(n_bins), include_lowest=True)
    data["time_bin"] = data["time_bin"].astype(int)

    # Average per (Magnet, Genotype, fly, time_bin)
    grouped = data.groupby([magnet_col, genotype_col, subject_col, "time_bin"])[value_col].mean().reset_index()
    grouped.rename(columns={value_col: f"avg_{value_col}"}, inplace=True)

    # Add bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))
    return grouped


def compute_permutation_tests_per_genotype(
    processed_data,
    metric="avg_distance_ball_0",
    magnet_col="Magnet",
    genotype_col="Genotype",
    control_group="n",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    results_per_genotype = {}
    genotypes = sorted(processed_data[genotype_col].unique())
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

    for geno in genotypes:
        sub = processed_data[processed_data[genotype_col] == geno]
        groups = sorted(sub[magnet_col].unique())
        if len(groups) != 2 or control_group not in groups:
            continue
        test_group = [g for g in groups if g != control_group][0]

        observed_diffs = []
        p_values = []

        iterator = tqdm(time_bins, desc=f"  {geno}: {test_group} vs {control_group}") if progress else time_bins
        for time_bin in iterator:
            bin_data = sub[sub["time_bin"] == time_bin]
            control_vals = bin_data[bin_data[magnet_col] == control_group][metric].values
            test_vals = bin_data[bin_data[magnet_col] == test_group][metric].values
            if len(control_vals) == 0 or len(test_vals) == 0:
                observed_diffs.append(np.nan)
                p_values.append(1.0)
                continue
            obs_diff = np.mean(test_vals) - np.mean(control_vals)
            observed_diffs.append(obs_diff)

            combined = np.concatenate([control_vals, test_vals])
            n_control = len(control_vals)
            perm_diffs = []
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_control = combined[:n_control]
                perm_test = combined[n_control:]
                perm_diffs.append(np.mean(perm_test) - np.mean(perm_control))
            p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
            p_values.append(p_value)

        p_values = np.array(p_values)
        significant_timepoints = np.where(p_values < alpha)[0]
        n_significant = int(np.sum(p_values < alpha))

        results_per_genotype[geno] = {
            "genotype": geno,
            "test_group": test_group,
            "control_group": control_group,
            "time_bins": time_bins,
            "observed_diffs": observed_diffs,
            "p_values": p_values,
            "significant_timepoints": significant_timepoints,
            "n_significant": n_significant,
        }
        print(f"    {geno}: Significant bins {n_significant}/{n_bins} (α={alpha})")

    return results_per_genotype


def create_trajectory_plots_by_genotype(
    data,
    processed,
    permutation_results_per_genotype,
    time_col="time",
    value_col="distance_ball_0",
    magnet_col="Magnet",
    genotype_col="Genotype",
    subject_col="fly",
    n_bins=12,
    output_dir=None,
):
    genotypes = sorted(data[genotype_col].unique())
    data = data.copy()
    # Convert raw distances to mm for plotting
    data[value_col] = data[value_col] / PIXELS_PER_MM

    # Downsample for plotting (every 290 frames)
    print("  Downsampling data for plotting...")
    data_ds = data.groupby(subject_col, group_keys=False).apply(lambda df: df.iloc[::290, :]).reset_index(drop=True)
    print(f"  Downsampled from {len(data)} to {len(data_ds)} points")

    # Figure: one subplot per genotype
    n_g = len(genotypes)
    fig, axes = plt.subplots(n_g, 1, figsize=(12, max(4, 3 * n_g)), sharex=True)
    if n_g == 1:
        axes = [axes]

    # Determine global y-range in mm for consistent annotation space
    # Use processed avg metric but convert to mm to match plotted units
    y_series_px = processed["avg_distance_ball_0"].to_numpy()
    y_series_mm = y_series_px / PIXELS_PER_MM
    y_min = float(np.nanmin(y_series_mm)) if y_series_mm.size else 0.0
    y_max = float(np.nanmax(y_series_mm)) if y_series_mm.size else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    # Bin info
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_width = (time_max - time_min) / n_bins

    # Helper: genotype-based color mapping (gray for Empty*, brown for DDC)
    def _genotype_color(g: str) -> str:
        gs = (g or "").lower()
        if "ddc" in gs:
            return "#8B4513"  # brown (DDC)
        if "empty" in gs:
            return "#7f7f7f"  # gray (Empty-Gal4 / EmptySplit)
        return "#7f7f7f"  # default to gray if unsure

    for i, geno in enumerate(genotypes):
        ax = axes[i]
        sub = data_ds[data_ds[genotype_col] == geno]
        n_control = sub[sub[magnet_col] == "n"][subject_col].nunique()
        n_test = sub[sub[magnet_col] == "y"][subject_col].nunique()

        # Color by genotype; style by magnet (y dotted, n solid)
        geno_color = _genotype_color(geno)

        # Plot control (n): solid thick line
        sub_n = sub[sub[magnet_col] == "n"].copy()
        if not sub_n.empty:
            sub_n["label"] = f"{geno} – Magnet n (n={n_control})"
            sns.lineplot(
                data=sub_n,
                x=time_col,
                y=value_col,
                color=geno_color,
                linestyle="-",
                linewidth=2.5,
                ax=ax,
                label=sub_n["label"].iloc[0],
            )

        # Plot magnet (y): dotted thinner line
        sub_y = sub[sub[magnet_col] == "y"].copy()
        if not sub_y.empty:
            sub_y["label"] = f"{geno} – Magnet y (n={n_test})"
            sns.lineplot(
                data=sub_y,
                x=time_col,
                y=value_col,
                color=geno_color,
                linestyle=":",
                linewidth=2.0,
                ax=ax,
                label=sub_y["label"].iloc[0],
            )

        # x-axis limits to match notebook style
        ax.set_xlim(3600, 7200)
        ax.set_ylim(y_min, y_max + 0.20 * y_range)
        y_max_extended = y_max + 0.20 * y_range

        # Annotate significance per bin
        results = permutation_results_per_genotype.get(geno)
        if results is not None:
            significant_bins = results["significant_timepoints"]
            for time_bin in range(n_bins):
                bin_start = time_min + time_bin * bin_width
                bin_end = bin_start + bin_width
                ax.axvline(bin_start, color="gray", linestyle="dotted", alpha=0.5)
                ax.axvline(bin_end, color="gray", linestyle="dotted", alpha=0.5)
                p_value = results["p_values"][time_bin]
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = ""
                if significance and time_bin in significant_bins:
                    y_star = y_max + 0.05 * y_range
                    ax.text(
                        bin_start + bin_width / 2,
                        y_star,
                        significance,
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        color="red",
                        fontweight="bold",
                    )
                y_pval = y_max + 0.11 * y_range
                p_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"
                p_color = "red" if time_bin in significant_bins else "gray"
                ax.text(
                    bin_start + bin_width / 2,
                    y_pval,
                    p_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=p_color,
                )

        ax.set_ylabel(f"{geno}: Ball distance (mm)", fontsize=12)
        # Legend: show genotype color once; include line style meaning
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc="lower right", fontsize=10, title=f"{geno} color")

    axes[-1].set_xlabel("Time", fontsize=12)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "trajectory_magnet_vs_control_by_genotype.pdf"
        png_path = output_dir / "trajectory_magnet_vs_control_by_genotype.png"
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"  Saved PDF: {pdf_path}")
        print(f"  Saved PNG: {png_path}")
    plt.close()


def generate_trajectory_plots(n_bins=12, n_permutations=10000, output_dir=None, show_progress=True):
    print("\n" + "=" * 60)
    print("GENERATING MAGNET × GENOTYPE TRAJECTORY PLOTS")
    print("=" * 60)
    df = load_coordinates_dataset()

    groups = sorted(df["Magnet"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected 2 Magnet groups, found {len(groups)}: {groups}")

    control_group = "n" if "n" in groups else groups[0]
    test_group = "y" if "y" in groups else groups[1]
    print(f"Magnet control: {control_group} | test: {test_group}")
    print(f"Genotypes: {sorted(df['Genotype'].unique())}")

    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        df,
        time_col="time",
        value_col="distance_ball_0",
        magnet_col="Magnet",
        genotype_col="Genotype",
        subject_col="fly",
        n_bins=n_bins,
    )
    print(f"Processed data shape: {processed.shape}")

    print(f"\nComputing per-genotype permutation tests ({n_permutations} permutations)...")
    perm_results = compute_permutation_tests_per_genotype(
        processed,
        metric="avg_distance_ball_0",
        magnet_col="Magnet",
        genotype_col="Genotype",
        control_group=control_group,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    print("\nGenerating plots...")
    out_dir = Path(output_dir) if output_dir else Path("/mnt/upramdya_data/MD/Magneblock_TNT/Plots/trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_trajectory_plots_by_genotype(
        df,
        processed,
        perm_results,
        time_col="time",
        value_col="distance_ball_0",
        magnet_col="Magnet",
        genotype_col="Genotype",
        subject_col="fly",
        n_bins=n_bins,
        output_dir=out_dir,
    )

    # Save statistical results summary
    stats_file = out_dir / "trajectory_permutation_statistics_by_genotype.txt"
    with open(stats_file, "w") as f:
        f.write("Magnet × Genotype Trajectory Permutation Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control group: {control_group}\n")
        f.write(f"Test group: {test_group}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Number of time bins: {n_bins}\n")
        f.write(f"Significance level: α = 0.05\n\n")
        for geno, results in perm_results.items():
            f.write(f"\nGenotype: {geno}\n")
            f.write(f"Significant bins: {results['n_significant']}/{n_bins}\n")
            f.write(f"Significant time bins: {list(results['significant_timepoints'])}\n\n")
            f.write("Bin | Obs. Diff | P-value | Significant\n")
            f.write("-" * 60 + "\n")
            for i, time_bin in enumerate(results["time_bins"]):
                obs_diff = results["observed_diffs"][i]
                p_val = results["p_values"][i]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                f.write(f"{time_bin:3d} | {obs_diff:9.3f} | {p_val:7.6f} | {sig}\n")
    print(f"\n✅ Statistical results saved to: {stats_file}")

    print("\n" + "=" * 60)
    print("✅ Trajectory plots by genotype generated successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Magnet vs Control trajectory plots per Genotype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-bins", type=int, default=12, help="Number of time bins (default: 12)")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/upramdya_data/MD/Magneblock_TNT/Plots/trajectories",
        help="Directory to save plots",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    generate_trajectory_plots(
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
