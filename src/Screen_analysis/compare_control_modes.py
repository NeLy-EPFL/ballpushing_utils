#!/usr/bin/env python3
"""
Standalone script to compare PCA results between different control modes.
Analyzes hits that are unique to each mode vs hits common to both.

Usage:
    python compare_control_modes.py <tailored_results_dir> <emptysplit_results_dir>
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def find_stats_file(results_dir, keyword="static"):
    """Find the statistics results CSV file in a results directory"""
    for file in Path(results_dir).rglob("*_stats_*.csv"):
        if keyword in str(file).lower():
            return file
    return None


def load_hits(stats_file, significance_col="Permutation_FDR_significant", genotype_col="Nickname"):
    """Load significant hits from a stats file"""
    if not stats_file or not os.path.exists(stats_file):
        return set(), None

    df = pd.read_csv(stats_file)

    # Try different significance column names
    sig_cols = [significance_col, "significant", "Permutation_pval"]
    sig_col_used = None

    for col in sig_cols:
        if col in df.columns:
            sig_col_used = col
            break

    if not sig_col_used:
        print(f"⚠️  No significance column found in {stats_file}")
        return set(), df

    # Filter significant hits
    if sig_col_used == "Permutation_pval":
        hits = set(df[df[sig_col_used] < 0.05][genotype_col])
    else:
        hits = set(df[df[sig_col_used] == True][genotype_col])

    return hits, df


def compare_control_modes(tailored_dir, emptysplit_dir, output_dir="comparison_tailored_vs_emptysplit"):
    """
    Compare results between tailored controls and Empty-Split control modes.
    Identifies hits that are unique to each mode and hits common to both.
    """
    print("📊 CONTROL MODE COMPARISON")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find stats files
    print(f"\n🔍 Looking for results files...")
    tailored_stats = find_stats_file(tailored_dir, "static")
    emptysplit_stats = find_stats_file(emptysplit_dir, "static")

    if not tailored_stats:
        print(f"❌ Could not find stats file in tailored directory: {tailored_dir}")
        return
    if not emptysplit_stats:
        print(f"❌ Could not find stats file in emptysplit directory: {emptysplit_dir}")
        return

    print(f"  ✅ Tailored:    {tailored_stats}")
    print(f"  ✅ Empty-Split: {emptysplit_stats}")

    # Load hits
    print(f"\n📈 Loading significant hits...")
    hits_tailored, df_tailored = load_hits(tailored_stats)
    hits_emptysplit, df_emptysplit = load_hits(emptysplit_stats)

    # Calculate overlaps
    hits_both = hits_tailored & hits_emptysplit
    hits_tailored_only = hits_tailored - hits_emptysplit
    hits_emptysplit_only = hits_emptysplit - hits_tailored
    all_genotypes = hits_tailored | hits_emptysplit

    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Hits with tailored controls:        {len(hits_tailored):4d}")
    print(f"Hits with Empty-Split control:      {len(hits_emptysplit):4d}")
    print(f"{'-'*70}")
    print(
        f"Hits in BOTH modes (robust):        {len(hits_both):4d}  ({len(hits_both)/max(len(all_genotypes), 1)*100:.1f}%)"
    )
    print(
        f"Hits ONLY with tailored controls:   {len(hits_tailored_only):4d}  ({len(hits_tailored_only)/max(len(all_genotypes), 1)*100:.1f}%)"
    )
    print(
        f"Hits ONLY with Empty-Split control: {len(hits_emptysplit_only):4d}  ({len(hits_emptysplit_only)/max(len(all_genotypes), 1)*100:.1f}%)"
    )
    print(f"{'='*70}")
    print(f"Total unique hits across both:      {len(all_genotypes):4d}")
    print(f"{'='*70}")

    # Create detailed comparison DataFrame
    comparison_data = []

    for genotype in sorted(all_genotypes):
        in_tailored = genotype in hits_tailored
        in_emptysplit = genotype in hits_emptysplit

        # Get p-values from both modes
        pval_tailored = 1.0
        pval_emptysplit = 1.0

        if df_tailored is not None:
            pval_t = df_tailored[df_tailored["Nickname"] == genotype]["Permutation_pval"].values
            if len(pval_t) > 0:
                pval_tailored = pval_t[0]

        if df_emptysplit is not None:
            pval_e = df_emptysplit[df_emptysplit["Nickname"] == genotype]["Permutation_pval"].values
            if len(pval_e) > 0:
                pval_emptysplit = pval_e[0]

        # Categorize
        if in_tailored and in_emptysplit:
            category = "Both (robust)"
        elif in_tailored:
            category = "Tailored only"
        else:
            category = "Empty-Split only"

        comparison_data.append(
            {
                "Genotype": genotype,
                "Category": category,
                "Hit_Tailored": in_tailored,
                "Hit_EmptySplit": in_emptysplit,
                "Pval_Tailored": pval_tailored,
                "Pval_EmptySplit": pval_emptysplit,
                "Pval_Difference": abs(pval_tailored - pval_emptysplit),
                "Min_Pval": min(pval_tailored, pval_emptysplit),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(["Category", "Min_Pval"])

    # Save comparison CSV
    csv_path = os.path.join(output_dir, "control_mode_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed comparison CSV: {csv_path}")

    # Create summary text file
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CONTROL MODE COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Tailored Controls Directory:   {tailored_dir}\n")
        f.write(f"Empty-Split Control Directory: {emptysplit_dir}\n\n")
        f.write(f"{'='*70}\n")
        f.write(f"Hits with tailored controls:        {len(hits_tailored):4d}\n")
        f.write(f"Hits with Empty-Split control:      {len(hits_emptysplit):4d}\n\n")
        f.write(
            f"Hits in BOTH modes (robust):        {len(hits_both):4d}  ({len(hits_both)/max(len(all_genotypes), 1)*100:.1f}%)\n"
        )
        f.write(
            f"Tailored ONLY:                      {len(hits_tailored_only):4d}  ({len(hits_tailored_only)/max(len(all_genotypes), 1)*100:.1f}%)\n"
        )
        f.write(
            f"Empty-Split ONLY:                   {len(hits_emptysplit_only):4d}  ({len(hits_emptysplit_only)/max(len(all_genotypes), 1)*100:.1f}%)\n"
        )
        f.write(f"{'='*70}\n\n")

        # Hits in both modes
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"HITS IN BOTH MODES (n={len(hits_both)}) - MOST ROBUST\n")
        f.write("=" * 70 + "\n")
        both_df = comparison_df[comparison_df["Category"] == "Both (robust)"].sort_values("Min_Pval")
        for _, row in both_df.iterrows():
            f.write(
                f"{row['Genotype']:35s}  p_tail={row['Pval_Tailored']:7.5f}  p_empty={row['Pval_EmptySplit']:7.5f}\n"
            )

        # Tailored only
        if hits_tailored_only:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"HITS ONLY WITH TAILORED CONTROLS (n={len(hits_tailored_only)})\n")
            f.write("=" * 70 + "\n")
            tail_df = comparison_df[comparison_df["Category"] == "Tailored only"].sort_values("Pval_Tailored")
            for _, row in tail_df.iterrows():
                f.write(
                    f"{row['Genotype']:35s}  p_tail={row['Pval_Tailored']:7.5f}  p_empty={row['Pval_EmptySplit']:7.5f}\n"
                )

        # Empty-Split only
        if hits_emptysplit_only:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"HITS ONLY WITH EMPTY-SPLIT CONTROL (n={len(hits_emptysplit_only)})\n")
            f.write("=" * 70 + "\n")
            empty_df = comparison_df[comparison_df["Category"] == "Empty-Split only"].sort_values("Pval_EmptySplit")
            for _, row in empty_df.iterrows():
                f.write(
                    f"{row['Genotype']:35s}  p_tail={row['Pval_Tailored']:7.5f}  p_empty={row['Pval_EmptySplit']:7.5f}\n"
                )

    print(f"💾 Summary text file: {summary_path}")

    # Create Venn diagram
    try:
        from matplotlib_venn import venn2

        plt.figure(figsize=(10, 8))
        venn = venn2(
            [hits_tailored, hits_emptysplit],
            set_labels=("Tailored Controls", "Empty-Split Control"),
        )

        # Add percentage labels
        if venn.get_label_by_id("10"):
            venn.get_label_by_id("10").set_text(
                f"{len(hits_tailored_only)}\n({len(hits_tailored_only)/max(len(all_genotypes), 1)*100:.1f}%)"
            )
        if venn.get_label_by_id("01"):
            venn.get_label_by_id("01").set_text(
                f"{len(hits_emptysplit_only)}\n({len(hits_emptysplit_only)/max(len(all_genotypes), 1)*100:.1f}%)"
            )
        if venn.get_label_by_id("11"):
            venn.get_label_by_id("11").set_text(
                f"{len(hits_both)}\n({len(hits_both)/max(len(all_genotypes), 1)*100:.1f}%)"
            )

        plt.title(
            f"Control Mode Comparison: Significant Hits\n(Total unique: {len(all_genotypes)})",
            fontsize=14,
            fontweight="bold",
        )

        venn_path = os.path.join(output_dir, "control_mode_venn.png")
        plt.savefig(venn_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"💾 Venn diagram: {venn_path}")
    except ImportError:
        print("⚠️  matplotlib_venn not available, skipping Venn diagram")
        print("    Install with: pip install matplotlib-venn")

    # Create comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    categories = ["Both modes\n(robust)", "Tailored\nonly", "Empty-Split\nonly"]
    counts = [len(hits_both), len(hits_tailored_only), len(hits_emptysplit_only)]
    colors = ["#2ecc71", "#3498db", "#e67e22"]  # Green, blue, orange

    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Number of Genotypes", fontsize=14, fontweight="bold")
    ax.set_title("Comparison of Hits by Control Mode", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add count and percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        percentage = count / max(len(all_genotypes), 1) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    bar_path = os.path.join(output_dir, "control_mode_comparison_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Bar plot: {bar_path}")

    # Create p-value comparison scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all genotypes
    for category, color, marker in [
        ("Both (robust)", "green", "o"),
        ("Tailored only", "blue", "^"),
        ("Empty-Split only", "orange", "s"),
    ]:
        subset = comparison_df[comparison_df["Category"] == category]
        if len(subset) > 0:
            ax.scatter(
                subset["Pval_Tailored"],
                subset["Pval_EmptySplit"],
                c=color,
                label=category,
                alpha=0.6,
                s=80,
                marker=marker,
                edgecolors="black",
            )

    # Add diagonal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=2, label="Equal p-values")

    # Add significance threshold lines
    ax.axhline(0.05, color="red", linestyle=":", alpha=0.5, linewidth=1.5, label="p=0.05 threshold")
    ax.axvline(0.05, color="red", linestyle=":", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("P-value (Tailored Controls)", fontsize=12, fontweight="bold")
    ax.set_ylabel("P-value (Empty-Split Control)", fontsize=12, fontweight="bold")
    ax.set_title("P-value Comparison Between Control Modes", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    scatter_path = os.path.join(output_dir, "pvalue_comparison_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Scatter plot: {scatter_path}")

    # Print detailed lists to console
    print(f"\n{'='*70}")
    print(f"🏆 HITS IN BOTH MODES ({len(hits_both)}) - MOST ROBUST:")
    print(f"{'='*70}")
    for genotype in sorted(hits_both)[:30]:
        row = comparison_df[comparison_df["Genotype"] == genotype].iloc[0]
        print(f"   {genotype:35s}  p_tail={row['Pval_Tailored']:.5f}  p_empty={row['Pval_EmptySplit']:.5f}")
    if len(hits_both) > 30:
        print(f"   ... and {len(hits_both) - 30} more (see {summary_path})")

    if hits_tailored_only:
        print(f"\n{'='*70}")
        print(f"🔵 HITS ONLY WITH TAILORED CONTROLS ({len(hits_tailored_only)}):")
        print(f"{'='*70}")
        for genotype in sorted(hits_tailored_only)[:20]:
            row = comparison_df[comparison_df["Genotype"] == genotype].iloc[0]
            print(f"   {genotype:35s}  p_tail={row['Pval_Tailored']:.5f}  p_empty={row['Pval_EmptySplit']:.5f}")
        if len(hits_tailored_only) > 20:
            print(f"   ... and {len(hits_tailored_only) - 20} more (see {summary_path})")

    if hits_emptysplit_only:
        print(f"\n{'='*70}")
        print(f"🟠 HITS ONLY WITH EMPTY-SPLIT CONTROL ({len(hits_emptysplit_only)}):")
        print(f"{'='*70}")
        for genotype in sorted(hits_emptysplit_only)[:20]:
            row = comparison_df[comparison_df["Genotype"] == genotype].iloc[0]
            print(f"   {genotype:35s}  p_tail={row['Pval_Tailored']:.5f}  p_empty={row['Pval_EmptySplit']:.5f}")
        if len(hits_emptysplit_only) > 20:
            print(f"   ... and {len(hits_emptysplit_only) - 20} more (see {summary_path})")

    print(f"\n{'='*70}")
    print(f"✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}/")
    print(f"  • CSV:        {csv_path}")
    print(f"  • Summary:    {summary_path}")
    print(f"  • Plots:      {output_dir}/*.png")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCA results between tailored and Empty-Split control modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_control_modes.py pca_results_tailored/ pca_results_emptysplit/
    python compare_control_modes.py pca_results_tailored/ pca_results_emptysplit/ -o my_comparison/
        """,
    )
    parser.add_argument("tailored_dir", help="Directory containing tailored control results")
    parser.add_argument("emptysplit_dir", help="Directory containing Empty-Split control results")
    parser.add_argument(
        "-o",
        "--output",
        default="comparison_tailored_vs_emptysplit",
        help="Output directory for comparison results (default: comparison_tailored_vs_emptysplit)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.tailored_dir):
        print(f"❌ Tailored directory not found: {args.tailored_dir}")
        return 1

    if not os.path.exists(args.emptysplit_dir):
        print(f"❌ Empty-Split directory not found: {args.emptysplit_dir}")
        return 1

    compare_control_modes(args.tailored_dir, args.emptysplit_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
