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

        # Build compact plot: 2 boxes per genotype (n then y), colors per Magnet
        # Prepare plotting data (converted for display)
        plot_rows = []
        for g in genotypes:
            sub = plot_df[plot_df[genotype_col] == g]
            if sub.empty:
                continue
            for mval, label in [("n", "Control"), ("y", "Magnet block")]:
                vals = sub[sub[magnet_col] == mval][metric].dropna()
                if len(vals) == 0:
                    continue
                vals = convert_metric_data(vals, metric)
                for v in vals:
                    plot_rows.append({genotype_col: g, magnet_col: mval, metric: float(v)})

        if not plot_rows:
            print("  ⚠️  No values for plotting, skipping")
            continue

        df_plot = pd.DataFrame(plot_rows)

        # Order: by genotype, within each genotype order n then y
        df_plot[genotype_col] = pd.Categorical(df_plot[genotype_col], categories=genotypes, ordered=True)
        df_plot[magnet_col] = pd.Categorical(df_plot[magnet_col], categories=["n", "y"], ordered=True)

        # Colors: genotype-based mapping; magnet differentiates within genotype by position
        # - Empty-Gal4/EmptySplit: gray
        # - DDC: brown
        # - Others: default to gray for consistency in DDC context
        def _colors_for_genotype(g):
            gs = (g or "").lower()
            if "ddc" in gs:
                return {"n": "#8B4513", "y": "#8B4513"}
            if "empty" in gs:
                return {"n": "#7f7f7f", "y": "#7f7f7f"}
            return {"n": "#7f7f7f", "y": "#7f7f7f"}

        # Figure sizing: narrow publication-style
        n_g = len(genotypes)
        fig_w = max(2.5, min(10, 1.4 * n_g + 1.0))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 3.5))

        # Positions: for each genotype allocate 2 positions (n,y) separated by small gap
        positions = []
        labels = []
        box_data = []
        colors = []

        xpos = 0
        for g in genotypes:
            cmap = _colors_for_genotype(g)
            g_vals_n = df_plot[(df_plot[genotype_col] == g) & (df_plot[magnet_col] == "n")][metric].values
            g_vals_y = df_plot[(df_plot[genotype_col] == g) & (df_plot[magnet_col] == "y")][metric].values
            if len(g_vals_n) == 0 and len(g_vals_y) == 0:
                # skip empty genotype
                continue
            # n box
            positions.append(xpos)
            labels.append(f"{g}\n(n)")
            box_data.append(g_vals_n)
            colors.append(cmap["n"])
            # y box
            positions.append(xpos + 1)
            labels.append(f"{g}\n(y)")
            box_data.append(g_vals_y)
            colors.append(cmap["y"])
            # gap between genotypes
            xpos += 3

        # Boxplot drawing
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=2, color="black"),
            boxprops=dict(linewidth=1.5, edgecolor="black"),
            whiskerprops=dict(linewidth=1.5, color="black"),
            capprops=dict(linewidth=1.5, color="black"),
        )

        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(colors[i])
            b.set_alpha(0.7)
            b.set_edgecolor("black")

        # Scatter points (jitter)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(box_data):
            if len(vals) == 0:
                continue
            x_jitter = rng.normal(positions[i], 0.08, size=len(vals))
            ax.scatter(x_jitter, vals, s=16, c=colors[i], alpha=0.6, edgecolors="none", zorder=3)

        # Add simple legend indicating color mapping per genotype
        # One patch per genotype with its color
        import matplotlib.patches as mpatches

        legend_patches = []
        added = set()
        for g in genotypes:
            if g in added:
                continue
            cmap = _colors_for_genotype(g)
            # use 'n' color as representative
            legend_patches.append(mpatches.Patch(color=cmap["n"], label=g))
            added.add(g)
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right", fontsize=8, title="Genotype colors")

        # Per-genotype significance bars
        # Map genotypes to x-range for their two boxes
        ymax = np.max([np.max(v) for v in box_data if len(v)]) if any(len(v) for v in box_data) else 1.0
        ymin = np.min([np.min(v) for v in box_data if len(v)]) if any(len(v) for v in box_data) else 0.0
        yr = ymax - ymin if ymax > ymin else 1.0

        # Bring corrected significance info into a dict
        sig_map = {(s["Genotype"],): (s["significant"], s.get("p_corrected", s["p_value"])) for s in per_genotype_stats}

        for gi, g in enumerate(genotypes):
            left = gi * 3  # positions for genotype block: left (n) and left+1 (y)
            # find stats for this genotype
            # some genotypes may have been skipped due to low n
            match = [s for s in per_genotype_stats if s["Genotype"] == g]
            if not match:
                continue
            sig = match[0]["significant"]
            pval = match[0].get("p_corrected", match[0]["p_value"])
            if pval < 0.001:
                sym = "***"
            elif pval < 0.01:
                sym = "**"
            elif pval < 0.05:
                sym = "*"
            else:
                sym = "ns"

            if sym != "ns":
                h = ymax + 0.08 * yr
                ax.plot([left, left + 1], [h, h], "k-", linewidth=1.3)
                ax.text((left + left + 1) / 2, h + 0.02 * yr, sym, ha="center", va="bottom", fontsize=11)

        # Axes formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel(format_metric_label(metric), fontsize=11)
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
