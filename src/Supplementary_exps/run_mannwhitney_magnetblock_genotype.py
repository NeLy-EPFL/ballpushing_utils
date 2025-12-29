#!/usr/bin/env python3
"""
Mann-Whitney U plots for Magnet (y/n) across Genotype (2+ groups).

For each metric, creates compact publication-style boxplots grouped by Genotype,
with two boxes per genotype (n=Control, y=Magnet block). Applies FDR correction
across the per-genotype comparisons for each metric. Colors follow F1 style:
- Control (n): Orange (#ff7f0e)
- Magnet (y): Blue (#1f77b4)

Usage:
    python run_mannwhitney_magnetblock_genotype.py [--no-overwrite] [--test]

Dataset:
    /mnt/upramdya_data/MD/Magneblock_TNT/Datasets/251205_13_summary_magnetblock_TNT_DDC_Data/summary/pooled_summary.feather
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import time

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def is_distance_metric(metric_name: str) -> bool:
    distance_keywords = [
        "distance",
        "dist",
        "head_ball",
        "fly_distance_moved",
        "max_distance",
        "distance_moved",
        "distance_ratio",
    ]
    return any(keyword in metric_name.lower() for keyword in distance_keywords)


def is_time_metric(metric_name: str) -> bool:
    time_keywords = ["time", "duration", "pause", "stop", "freeze", "chamber_exit_time", "time_chamber_beginning"]
    if "ratio" in metric_name.lower():
        return False
    return any(keyword in metric_name.lower() for keyword in time_keywords)


def convert_metric_data(series: pd.Series, metric_name: str) -> pd.Series:
    if is_distance_metric(metric_name):
        return series / PIXELS_PER_MM
    if is_time_metric(metric_name):
        return series / 60.0
    return series


def format_metric_label(metric_name: str) -> str:
    unit = "(mm)" if is_distance_metric(metric_name) else "(min)" if is_time_metric(metric_name) else ""
    if unit:
        return f"{metric_name} {unit}"
    return metric_name


def _normalize_magnet(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"n", "no", "false", "0"}:
        return "n"
    if s in {"y", "yes", "true", "1"}:
        return "y"
    return s


def _normalize_genotype_str(s):
    if pd.isna(s):
        return np.nan
    return " ".join(str(s).strip().split())


def load_and_clean_dataset(test_mode=False, test_sample_size=200) -> pd.DataFrame:
    dataset_path = "/mnt/upramdya_data/MD/Magneblock_TNT/Datasets/251205_13_summary_magnetblock_TNT_DDC_Data/summary/pooled_summary.feather"

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"✅ Loaded dataset with shape {df.shape}")

    # Required columns
    required = ["Magnet", "Genotype"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in dataset. Columns: {list(df.columns)}")

    # Normalize columns
    df = df.copy()
    df["Magnet"] = df["Magnet"].apply(_normalize_magnet)
    df["Genotype"] = df["Genotype"].apply(_normalize_genotype_str)

    # Report groups
    print(f"Magnet groups: {sorted([g for g in df['Magnet'].dropna().unique()])}")
    print(f"Genotypes: {sorted([g for g in df['Genotype'].dropna().unique()])}")

    # Convert bool columns to int to avoid plotting issues
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Optionally sample for test mode
    if test_mode and len(df) > test_sample_size:
        df = df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: Sampled {len(df)} rows")

    return df


def generate_magnet_genotype_plots(
    data: pd.DataFrame,
    metrics,
    magnet_col="Magnet",
    genotype_col="Genotype",
    control_magnet="n",
    output_dir="/mnt/upramdya_data/MD/Magneblock_TNT/Plots/MagnetGenotype_MannWhitney",
    alpha=0.05,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Publication-friendly fonts
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Ensure only valid Magnet values
    valid = data[data[magnet_col].isin(["n", "y"])].copy()
    if valid.empty:
        print("No valid Magnet values ('n'/'y') after normalization.")
        return pd.DataFrame()

    genotypes = sorted(valid[genotype_col].dropna().unique())
    if len(genotypes) < 1:
        print("No genotypes found.")
        return pd.DataFrame()

    all_stats = []

    for idx, metric in enumerate(metrics, 1):
        t0 = time.time()
        print(f"[{idx}/{len(metrics)}] Analyzing metric: {metric}")

        # Drop rows with NaN in metric
        plot_df = valid.dropna(subset=[metric]).copy()
        if plot_df.empty:
            print("  ⚠️  No data for metric, skipping")
            continue

        # Per-genotype MW tests (y vs n)
        per_genotype_stats = []
        pvals = []
        testable_genotypes = []

        for g in genotypes:
            sub = plot_df[plot_df[genotype_col] == g]
            g_n = sub[sub[magnet_col] == "n"][metric].dropna()
            g_y = sub[sub[magnet_col] == "y"][metric].dropna()

            if len(g_n) < 3 or len(g_y) < 3:
                continue

            # MW on raw values
            u, p = mannwhitneyu(g_y, g_n, alternative="two-sided")

            # Effect summaries on converted values for display
            g_n_conv = convert_metric_data(g_n, metric)
            g_y_conv = convert_metric_data(g_y, metric)
            med_n = float(g_n_conv.median())
            med_y = float(g_y_conv.median())

            per_genotype_stats.append(
                {
                    "Metric": metric,
                    "Genotype": g,
                    "Control": control_magnet,
                    "Test": "y",
                    "n_control": len(g_n),
                    "n_test": len(g_y),
                    "median_control": med_n,
                    "median_test": med_y,
                    "median_diff": med_y - med_n,
                    "u_stat": float(u),
                    "p_value": float(p),
                }
            )
            pvals.append(p)
            testable_genotypes.append(g)

        if not per_genotype_stats:
            print("  ⚠️  Not enough data across genotypes, skipping plot")
            continue

        # FDR across genotypes for this metric
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for i, g in enumerate(testable_genotypes):
            per_genotype_stats[i]["p_corrected"] = float(p_adj[i])
            per_genotype_stats[i]["significant"] = bool(reject[i])

        all_stats.extend(per_genotype_stats)

        # Build F1-style plot: Use combined grouping with genotype color and magnet as fill/no-fill
        # Prepare plotting data (converted for display)
        plot_rows = []
        for g in genotypes:
            sub = plot_df[plot_df[genotype_col] == g]
            if sub.empty:
                continue
            for mval in ["n", "y"]:
                vals = sub[sub[magnet_col] == mval][metric].dropna()
                if len(vals) == 0:
                    continue
                vals = convert_metric_data(vals, metric)
                for v in vals:
                    plot_rows.append(
                        {genotype_col: g, magnet_col: mval, metric: float(v), "combined_group": f"{g} + {mval}"}
                    )

        if not plot_rows:
            print("  ⚠️  No values for plotting, skipping")
            continue

        df_plot = pd.DataFrame(plot_rows)

        # Create combined groups for plotting
        unique_groups = sorted(df_plot["combined_group"].unique())

        # Colors: genotype-based mapping (consistent across magnet conditions)
        # - Empty-Gal4/EmptySplit: gray
        # - DDC: brown
        # - Others: default to gray for consistency in DDC context
        def _color_for_genotype(g):
            gs = (g or "").lower()
            if "ddc" in gs:
                return "#8B4513"
            if "empty" in gs:
                return "#7f7f7f"
            return "#7f7f7f"

        # Map colors to each combined group based on genotype
        genotype_colors = {}
        for g in genotypes:
            genotype_colors[g] = _color_for_genotype(g)

        colors = []
        for group in unique_groups:
            genotype = group.split(" + ")[0]
            colors.append(genotype_colors[genotype])

        # Figure sizing: narrow publication-style
        n_groups = len(unique_groups)
        fig_w = max(4, min(12, 0.8 * n_groups + 2.0))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.5))

        # Add scatter with jitter FIRST (so boxplot appears on top)
        for i, group in enumerate(unique_groups):
            group_data = df_plot[df_plot["combined_group"] == group][metric].values
            x_pos = np.random.normal(i, 0.04, size=len(group_data))
            ax.scatter(
                x_pos,
                group_data,
                alpha=0.6,
                s=50,
                color=colors[i],
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )

        # Create boxplot (drawn on top)
        box_plot = sns.boxplot(
            data=df_plot,
            x="combined_group",
            y=metric,
            ax=ax,
            order=unique_groups,
            palette=colors,
            width=0.6,
            zorder=1,
        )

        # F1-style: filled boxes for magnet=y, white boxes for magnet=n
        for i, (patch, group) in enumerate(zip(box_plot.patches, unique_groups)):
            magnet = group.split(" + ")[1]
            if magnet == "n":
                # Control (no magnet): white fill, colored edge
                patch.set_facecolor("white")
                patch.set_edgecolor(colors[i])
                patch.set_linewidth(2.5)
            else:
                # Magnet block: colored fill, black edge
                patch.set_facecolor(colors[i])
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)

        # Style whiskers and medians to match box styling
        for i, group in enumerate(unique_groups):
            magnet = group.split(" + ")[1]
            color = colors[i]
            for j in range(i * 6, (i + 1) * 6):
                if j < len(ax.lines):
                    line = ax.lines[j]
                    if magnet == "n":
                        line.set_color(color)
                        line.set_linewidth(2)
                    else:
                        line.set_color("black")
                        line.set_linewidth(1.5)

        # Per-genotype significance bars (connect n vs y within same genotype)
        ymax = df_plot[metric].max()
        ymin = df_plot[metric].min()
        yr = ymax - ymin if ymax > ymin else 1.0

        # Track bar positions to avoid overlaps
        bar_y_positions = []

        for gi, g in enumerate(genotypes):
            # Find indices of n and y boxes for this genotype
            group_n = f"{g} + n"
            group_y = f"{g} + y"

            if group_n not in unique_groups or group_y not in unique_groups:
                continue

            idx_n = unique_groups.index(group_n)
            idx_y = unique_groups.index(group_y)

            # Find stats for this genotype
            match = [s for s in per_genotype_stats if s["Genotype"] == g]
            if not match:
                continue

            sig = match[0]["significant"]
            pval = match[0].get("p_corrected", match[0]["p_value"])

            if pval < 0.001:
                p_text = "p < 0.001"
                stars = "***"
            elif pval < 0.01:
                p_text = f"p = {pval:.3f}"
                stars = "**"
            elif pval < 0.05:
                p_text = f"p = {pval:.3f}"
                stars = "*"
            else:
                p_text = f"p = {pval:.3f}"
                stars = "ns"

            # Find a good y position for this bar (avoid overlaps)
            max_overlap_level = 0
            for existing_i, existing_j, level in bar_y_positions:
                if not (idx_y < existing_i or idx_n > existing_j):
                    max_overlap_level = max(max_overlap_level, level + 1)

            bar_level = max_overlap_level
            y_bar = ymax + yr * (0.05 + bar_level * 0.08)
            bar_y_positions.append((idx_n, idx_y, bar_level))
            bar_height = yr * 0.02

            # Draw bar
            ax.plot([idx_n, idx_y], [y_bar, y_bar], "k-", linewidth=1.5)
            ax.plot([idx_n, idx_n], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)
            ax.plot([idx_y, idx_y], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)

            # Add p-value text (always show, bold if significant)
            if sig:
                full_text = f"{stars} {p_text}"
                fontweight = "bold"
            else:
                full_text = p_text
                fontweight = "normal"

            ax.text(
                (idx_n + idx_y) / 2,
                y_bar + bar_height,
                full_text,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight=fontweight,
                fontname="Arial",
            )

        # Adjust y-axis limits to accommodate bars
        if bar_y_positions:
            max_bar_level = max((level for _, _, level in bar_y_positions), default=0)
            ax.set_ylim(ymin - 0.05 * yr, ymax + yr * (0.15 + max_bar_level * 0.08))

        # Axes formatting with improved labels
        x_labels = []
        for group in unique_groups:
            genotype, magnet = group.split(" + ")
            magnet_label = "Control" if magnet == "n" else "Magnet"
            n = len(df_plot[df_plot["combined_group"] == group])
            x_labels.append(f"{genotype}\n{magnet_label}\n(n={n})")

        ax.set_xticks(range(len(unique_groups)))
        ax.set_xticklabels(x_labels, fontsize=9, fontname="Arial", rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("Genotype + Magnet Condition", fontsize=11, fontname="Arial")
        ax.set_ylabel(format_metric_label(metric), fontsize=11, fontname="Arial")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        # Save
        safe = metric.replace("/", "_").replace(" ", "_")
        out_pdf = output_dir / f"{safe}_magnet_genotype.pdf"
        out_png = output_dir / f"{safe}_magnet_genotype.png"
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: {out_pdf} (took {time.time() - t0:.1f}s)")

    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_path = output_dir / "magnet_genotype_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✅ Statistics saved to: {stats_path}")
    else:
        print("\n⚠️  No statistics computed")

    return stats_df


def main(overwrite=True, test_mode=False):
    print("\n" + "=" * 80)
    print("MAGNET × GENOTYPE MANN-WHITNEY U TEST ANALYSIS")
    print("=" * 80 + "\n")

    df = load_and_clean_dataset(test_mode=test_mode)

    # Output directory under /Volumes (requested)
    output_dir = Path("/mnt/upramdya_data/MD/Magneblock_TNT/Plots/MagnetGenotype_MannWhitney")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Select numeric metrics excluding grouping columns and obvious identifiers
    exclude_cols = {"Magnet", "Genotype", "fly", "Corridor"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [c for c in numeric_cols if c not in exclude_cols]
    print(f"Found {len(metrics)} numeric metrics")

    if test_mode:
        metrics = metrics[:3]
        print(f"🧪 TEST MODE: Limiting to {len(metrics)} metrics: {metrics}")

    if not metrics:
        print("No metrics to analyze. Exiting.")
        return

    _ = generate_magnet_genotype_plots(
        data=df,
        metrics=metrics,
        magnet_col="Magnet",
        genotype_col="Genotype",
        control_magnet="n",
        output_dir=output_dir,
        alpha=0.05,
    )

    print("\n" + "=" * 80)
    print("✅ MAGNET × GENOTYPE ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U plots for Magnet (y/n) across Genotype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reserved for parity with other scripts (plots always overwritten here)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process first 3 metrics only",
    )
    args = parser.parse_args()
    main(overwrite=not args.no_overwrite, test_mode=args.test)
