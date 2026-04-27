#!/usr/bin/env python3
"""
Visual comparison of Tailored vs Empty-Split control modes.

- Loads enhanced_consistency_scores.csv from each results directory
- Builds a side-by-side dot plot of consistency by genotype and control mode
- Colors labels by brain region (using Config registries when available)
- Highlights robust hits (>= 80% Combined_Consistency) in each mode and both
- Saves figure and a merged CSV

Usage:
  python compare_control_modes_visual.py --tailored-dir <dir> --emptysplit-dir <dir> [--output-dir comparison_tailored_vs_emptysplit]
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Attempt to load color registry from Config
def load_region_colors():
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import Config  # type: ignore

        registries = Config.registries
        split = Config.SplitRegistry
        color_dict = Config.color_dict
        nickname_to_br = dict(zip(split["Nickname"], split["Simplified region"]))
        return nickname_to_br, color_dict
    except Exception as e:
        print(f"⚠️  Could not load brain region registry/colors from Config: {e}")
        return {}, {}


def parse_args():
    p = argparse.ArgumentParser(description="Visual comparison of tailored vs emptysplit control modes")
    p.add_argument("--tailored-dir", required=True, help="Results directory for tailored mode")
    p.add_argument("--emptysplit-dir", required=True, help="Results directory for emptysplit mode")
    p.add_argument(
        "--output-dir",
        default="comparison_tailored_vs_emptysplit",
        help="Directory to save comparison outputs",
    )
    p.add_argument(
        "--robust-threshold",
        type=float,
        default=0.80,
        help="Consistency threshold to mark as robust (Combined_Consistency)",
    )
    p.add_argument(
        "--metric",
        choices=["combined", "optimized"],
        default="combined",
        help="Which consistency metric to visualize and compare: combined or optimized (default: combined)",
    )
    return p.parse_args()


def load_consistency(df_path: str, which: str = "combined") -> pd.DataFrame:
    # Accept either results root or organized data_files location
    if not os.path.exists(df_path):
        alt_path = os.path.join(os.path.dirname(df_path), "data_files", os.path.basename(df_path))
        if os.path.exists(alt_path):
            df_path = alt_path
        else:
            raise FileNotFoundError(df_path)
    df = pd.read_csv(df_path)
    # Choose metric column with graceful fallbacks across schema versions
    if which == "combined":
        if "Combined_Consistency" in df.columns:
            col = "Combined_Consistency"
        elif "Overall_Consistency" in df.columns:
            col = "Overall_Consistency"
        elif {"Total_Hit_Count", "Total_Configs"}.issubset(df.columns):
            df["__combined_tmp__"] = df["Total_Hit_Count"].astype(float) / df["Total_Configs"].astype(float)
            col = "__combined_tmp__"
        else:
            raise ValueError("No combined consistency column found")
    else:  # optimized
        if "Optimized_Only_Consistency" in df.columns:
            col = "Optimized_Only_Consistency"
        elif "Optimized_Consistency" in df.columns:
            col = "Optimized_Consistency"
        elif {"Optimized_Hit_Count", "Optimized_Configs"}.issubset(df.columns):
            df["__optimized_tmp__"] = df["Optimized_Hit_Count"].astype(float) / df["Optimized_Configs"].astype(float)
            col = "__optimized_tmp__"
        else:
            raise ValueError("No optimized consistency column found")

    return df[["Genotype", col]].rename(columns={col: "Consistency"})


def merge_modes(tailored_dir: str, emptysplit_dir: str, which: str) -> pd.DataFrame:
    t_path = os.path.join(tailored_dir, "enhanced_consistency_scores.csv")
    e_path = os.path.join(emptysplit_dir, "enhanced_consistency_scores.csv")
    t = load_consistency(t_path, which)
    e = load_consistency(e_path, which)
    t["Mode"] = "Tailored"
    e["Mode"] = "Empty-Split"
    merged = pd.concat([t, e], ignore_index=True)
    # Pivot for robust flags computing and for CSV export
    pivot = merged.pivot_table(index="Genotype", columns="Mode", values="Consistency", fill_value=0)
    pivot = pivot.reset_index()
    return merged, pivot


def build_categories(pivot: pd.DataFrame, robust_threshold: float) -> pd.DataFrame:
    pivot = pivot.copy()
    # Exclude duplicated/irrelevant genotypes
    pivot = pivot[
        ~pivot["Genotype"].isin(
            ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "MB247-Gal4", "854 (OK107-Gal4)", "7362 (C739-Gal4)", "TNTxPR"]
        )
    ]
    if "Tailored" not in pivot.columns:
        pivot["Tailored"] = 0.0
    if "Empty-Split" not in pivot.columns:
        pivot["Empty-Split"] = 0.0
    pivot["Robust_Tailored"] = pivot["Tailored"] >= robust_threshold
    pivot["Robust_EmptySplit"] = pivot["Empty-Split"] >= robust_threshold

    def cat_row(r):
        if r["Robust_Tailored"] and r["Robust_EmptySplit"]:
            return "both"
        if r["Robust_Tailored"] and not r["Robust_EmptySplit"]:
            return "tailored_only"
        if r["Robust_EmptySplit"] and not r["Robust_Tailored"]:
            return "emptysplit_only"
        return "none"

    pivot["Category"] = pivot.apply(cat_row, axis=1)
    return pivot


def export_comparison_csv(pivot_with_cats: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    export = pivot_with_cats[["Genotype"]].copy()
    export["Consistency_Tailored"] = pivot_with_cats.get("Tailored", pd.Series(0, index=pivot_with_cats.index))
    export["Consistency_EmptySplit"] = pivot_with_cats.get("Empty-Split", pd.Series(0, index=pivot_with_cats.index))
    export["Robust_Tailored"] = pivot_with_cats["Robust_Tailored"]
    export["Robust_EmptySplit"] = pivot_with_cats["Robust_EmptySplit"]
    export["Category"] = pivot_with_cats["Category"]
    out_csv = os.path.join(out_dir, "control_mode_consistency_comparison.csv")
    export.to_csv(out_csv, index=False)
    print(f"💾 Saved merged comparison CSV: {out_csv}")


def plot_two_panel_bars(pivot_with_cats: pd.DataFrame, out_dir: str, robust_threshold: float, which: str):
    os.makedirs(out_dir, exist_ok=True)

    # Colors for background shading / bars
    cat_color = {
        "both": "#cfeadf",  # pale green
        "tailored_only": "#d6e3f5",  # pale blue
        "emptysplit_only": "#fde3bf",  # pale orange
        "none": "#efefef",
    }
    bar_edge = {
        "both": "#009E73",
        "tailored_only": "#0072B2",
        "emptysplit_only": "#E69F00",
        "none": "#666666",
    }

    # Prepare data for each mode
    t_df = pivot_with_cats[["Genotype", "Tailored", "Category"]].rename(columns={"Tailored": "Consistency"}).copy()
    e_df = (
        pivot_with_cats[["Genotype", "Empty-Split", "Category"]].rename(columns={"Empty-Split": "Consistency"}).copy()
    )

    # Ranking helper: primary sort by consistency DESC, then highlight priority among ties
    def sort_panel(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        d = df.copy()
        # Highlight priority: both (0) > mode-only (1) > none (2)
        if mode == "tailored":
            cat_priority_map = {"both": 0, "tailored_only": 1, "emptysplit_only": 1, "none": 2}
        else:
            cat_priority_map = {"both": 0, "emptysplit_only": 1, "tailored_only": 1, "none": 2}
        d["cat_priority"] = d["Category"].map(cat_priority_map).fillna(3)
        # Sort: consistency desc, then category priority, then name
        d = d.sort_values(
            by=["Consistency", "cat_priority", "Genotype"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        return d

    t_df = sort_panel(t_df, mode="tailored")
    e_df = sort_panel(e_df, mode="emptysplit")

    # Plot
    h = max(8, int(max(len(t_df), len(e_df)) * 0.16))
    fig, axes = plt.subplots(1, 2, figsize=(14, h), sharex=True)

    for ax, df, title in (
        (axes[0], t_df, "Tailored Controls"),
        (axes[1], e_df, "Empty-Split Control"),
    ):
        # Background shading per row by category
        for i, cat in enumerate(df["Category"].tolist()):
            ax.axhspan(i - 0.5, i + 0.5, color=cat_color.get(cat, "#efefef"), alpha=0.6, zorder=0)

        # Bars
        ax.barh(
            y=np.arange(len(df)),
            width=df["Consistency"].values,
            color="#444444",
            edgecolor=[bar_edge.get(c, "#666666") for c in df["Category"]],
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )

        # Genotype labels
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df["Genotype"].tolist(), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Optimized Consistency" if which == "optimized" else "Combined Consistency")
        ax.set_title(title)
        # Robustness threshold line
        ax.axvline(robust_threshold, color="#333333", linestyle="--", linewidth=1.2)
        ax.grid(axis="x", linestyle=":", alpha=0.3)

    # Unified legend explanation
    import matplotlib.patches as mpatches

    leg_patches = [
        mpatches.Patch(facecolor=cat_color["both"], edgecolor=bar_edge["both"], label="Robust in Both (≥80%)"),
        mpatches.Patch(
            facecolor=cat_color["tailored_only"], edgecolor=bar_edge["tailored_only"], label="Robust Tailored Only"
        ),
        mpatches.Patch(
            facecolor=cat_color["emptysplit_only"],
            edgecolor=bar_edge["emptysplit_only"],
            label="Robust Empty-Split Only",
        ),
        mpatches.Patch(facecolor=cat_color["none"], edgecolor=bar_edge["none"], label="Not Robust"),
    ]
    axes[1].legend(handles=leg_patches, loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(
        "Consistency by Control Mode (Two-Panel)\nShading = hit category, dashed line = 80% threshold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = os.path.join(out_dir, "control_mode_consistency_bars.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved two-panel bar figure: {out_png}")


def has_edge_cases(results_dir: str) -> bool:
    # Determine if any Edge_Case_Configs > 0 in the results CSV
    path = os.path.join(results_dir, "enhanced_consistency_scores.csv")
    if not os.path.exists(path):
        path = os.path.join(results_dir, "data_files", "enhanced_consistency_scores.csv")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        if "Edge_Case_Configs" in df.columns:
            return bool(df["Edge_Case_Configs"].fillna(0).astype(float).sum() > 0)
    except Exception:
        pass
    return False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("📊 Comparing control modes (visual)")
    print(f"  • Tailored:    {args.tailored_dir}")
    print(f"  • Empty-Split: {args.emptysplit_dir}")
    merged, pivot = merge_modes(args.tailored_dir, args.emptysplit_dir, args.metric)

    # Informative note when combined metric is requested but no edge cases exist
    if args.metric == "combined":
        t_has_edges = has_edge_cases(args.tailored_dir)
        e_has_edges = has_edge_cases(args.emptysplit_dir)
        if not t_has_edges and not e_has_edges:
            print("⚠️  No edge cases detected in either directory; Combined == Optimized for both.")
    pivot_with_cats = build_categories(pivot, args.robust_threshold)
    export_comparison_csv(pivot_with_cats, args.output_dir)
    plot_two_panel_bars(pivot_with_cats, args.output_dir, args.robust_threshold, args.metric)
    print("✅ Visual comparison complete")


if __name__ == "__main__":
    main()
