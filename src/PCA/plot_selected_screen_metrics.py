#!/usr/bin/env python3
"""
Generate selected summary-metric plots for selected TNT screen genotypes.

For each requested genotype/nickname and each requested metric:
- compares target vs control
- performs a two-sided permutation test on mean difference
- generates boxplot + jittered scatter plot
- saves outputs in: OUTPUT_ROOT / BrainRegion / Nickname /

Examples
--------
python plot_selected_screen_metrics.py \
  --metrics first_major_event flailing max_event \
  --genotypes TNTxLC16-1

python plot_selected_screen_metrics.py \
  --metrics first_major_event,flailing,max_event \
  --genotypes LC16-1 TNTxLC10-2 \
  --control-mode emptysplit \
  --output-root /mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics

python plot_selected_screen_metrics.py \
    --genotypes LC16-1 TNTxLC10-2
    # if --metrics is omitted, all available numeric metrics are used
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import Config


# Illustrator-editable text
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


DEFAULT_DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
DEFAULT_OUTPUT_ROOT = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics"

CONTROL_NICKNAMES = {
    "y": "Empty-Split",
    "n": "Empty-Gal4",
    "m": "TNTxPR",
}

METADATA_COLUMNS_FOR_METRICS = {
    "Nickname",
    "Brain region",
    "Simplified Nickname",
    "Simplified region",
    "Split",
    "Driver",
    "Experiment",
    "Date",
    "date",
    "Arena",
    "arena",
    "BallType",
    "Dissected",
    "Genotype",
    "Pretraining",
    "pretraining",
    "fly",
    "filename",
    "video",
    "path",
    "folder",
    "index",
}


def parse_list_args(values):
    items = []
    for value in values:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        items.extend(parts)
    return list(dict.fromkeys(items))


def format_p_value(p_value):
    if p_value is None or np.isnan(p_value):
        return "n/a"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.6f}"


def significance_label(p_value):
    if p_value is None or np.isnan(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def permutation_test_1d(group1, group2, n_permutations=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    observed = np.abs(np.mean(group1) - np.mean(group2))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        stat = np.abs(np.mean(perm[:n1]) - np.mean(perm[n1:]))
        if stat >= observed:
            count += 1

    return (count + 1) / (n_permutations + 1)


def load_dataset(data_path):
    df = pd.read_feather(data_path)
    df = Config.cleanup_data(df)

    # Convert boolean metrics to integers for consistent numeric handling
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Add mapping columns from split registry when available
    registry_cols = ["Nickname", "Simplified Nickname", "Simplified region", "Split"]
    reg = Config.SplitRegistry[[c for c in registry_cols if c in Config.SplitRegistry.columns]].copy()
    if "Nickname" in reg.columns:
        reg = reg[~reg["Nickname"].duplicated(keep="first")]

    if "Simplified Nickname" not in df.columns or "Simplified region" not in df.columns or "Split" not in df.columns:
        if "Nickname" in df.columns:
            df = df.merge(reg, on="Nickname", how="left", suffixes=("", "_reg"))
            for c in ["Simplified Nickname", "Simplified region", "Split"]:
                reg_col = f"{c}_reg"
                if reg_col in df.columns:
                    if c not in df.columns:
                        df[c] = df[reg_col]
                    else:
                        df[c] = df[c].fillna(df[reg_col])
                    df = df.drop(columns=[reg_col])

    return df


def auto_detect_metrics(df):
    """Auto-detect available numeric metrics while excluding metadata columns."""
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    metrics = [
        col for col in numeric_cols if col not in METADATA_COLUMNS_FOR_METRICS and not col.lower().startswith("unnamed")
    ]

    # Exclude obvious indexing/frame-like fields when numeric
    metrics = [col for col in metrics if col.lower() not in {"frame", "time", "id"}]

    return sorted(metrics)


def resolve_to_nickname(target, df):
    target = str(target).strip()
    if target in set(df["Nickname"].dropna().unique()):
        return target

    if "Simplified Nickname" in df.columns:
        match = df[df["Simplified Nickname"] == target]["Nickname"].dropna().unique()
        if len(match) == 1:
            return match[0]

    if "Genotype" in df.columns:
        match = df[df["Genotype"] == target]["Nickname"].dropna().unique()
        if len(match) == 1:
            return match[0]

    return None


def get_control_for_nickname(nickname, control_mode):
    row = Config.SplitRegistry[Config.SplitRegistry["Nickname"] == nickname]
    if row.empty:
        return None

    split_value = row["Split"].iloc[0]

    if control_mode == "tnt_pr":
        return "TNTxPR"

    if control_mode == "emptysplit":
        # Keep mutant logic aligned with existing PCA pipeline behavior
        return "TNTxPR" if split_value == "m" else "Empty-Split"

    # tailored
    return CONTROL_NICKNAMES.get(split_value)


def brain_region_for_nickname(df, nickname):
    row = df[df["Nickname"] == nickname]
    if row.empty:
        return "Unknown"
    if "Simplified region" in row.columns and row["Simplified region"].notna().any():
        return str(row["Simplified region"].dropna().iloc[0])
    if "Brain region" in row.columns and row["Brain region"].notna().any():
        return str(row["Brain region"].dropna().iloc[0])
    return "Unknown"


def display_name_for_nickname(df, nickname):
    row = df[df["Nickname"] == nickname]
    if row.empty:
        return nickname
    if "Simplified Nickname" in row.columns and row["Simplified Nickname"].notna().any():
        return str(row["Simplified Nickname"].dropna().iloc[0])
    return nickname


def sanitize_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def region_color(region_name):
    return Config.color_dict.get(str(region_name), "#7f7f7f")


def plot_metric_box_jitter(
    metric,
    control_name,
    target_name,
    control_values,
    target_values,
    output_dir,
    n_permutations,
    control_color,
    target_color,
):
    fig, ax = plt.subplots(figsize=(3.0, 1.6))

    box_data = [control_values, target_values]
    positions = [0, 1]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.2, edgecolor="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
        medianprops=dict(linewidth=1.5, color="black"),
    )

    bp["boxes"][0].set_facecolor("none")
    bp["boxes"][0].set_edgecolor("black")
    bp["boxes"][0].set_linewidth(1.2)
    bp["boxes"][1].set_facecolor("none")
    bp["boxes"][1].set_edgecolor("black")
    bp["boxes"][1].set_linewidth(1.2)

    jitter = 0.08
    x0 = np.random.normal(0, jitter, size=len(control_values))
    x1 = np.random.normal(1, jitter, size=len(target_values))
    ax.scatter(x0, control_values, s=10, alpha=0.75, c=control_color, edgecolors="black", linewidths=0.3)
    ax.scatter(x1, target_values, s=10, alpha=0.75, c=target_color, edgecolors="black", linewidths=0.3)

    p_value = permutation_test_1d(control_values, target_values, n_permutations=n_permutations, random_state=42)
    sig = significance_label(p_value)

    y_max = max(np.max(control_values), np.max(target_values))
    y_min = min(np.min(control_values), np.min(target_values))
    y_range = max(y_max - y_min, 1e-9)

    bar_height = y_max + 0.08 * y_range
    ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.2)
    ax.text(0.5, bar_height + 0.02 * y_range, sig, ha="center", va="bottom", fontsize=7, fontweight="bold")

    p_text = f"p = {format_p_value(p_value)}"
    ax.text(
        0.02,
        0.98,
        p_text,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=0.8),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"{control_name}\n(n={len(control_values)})", f"{target_name}\n(n={len(target_values)})"],
        fontsize=6,
    )
    ax.set_ylabel(metric, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=1.0)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    plt.tight_layout()

    metric_safe = sanitize_name(metric)
    pdf_path = output_dir / f"{metric_safe}.pdf"
    png_path = output_dir / f"{metric_safe}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    d_value = cohens_d(target_values, control_values)
    return {
        "metric": metric,
        "n_control": len(control_values),
        "n_target": len(target_values),
        "control_mean": float(np.mean(control_values)),
        "target_mean": float(np.mean(target_values)),
        "delta_target_minus_control": float(np.mean(target_values) - np.mean(control_values)),
        "cohens_d": float(d_value) if not np.isnan(d_value) else np.nan,
        "p_value": float(p_value),
        "significance": sig,
        "plot_pdf": str(pdf_path),
        "plot_png": str(png_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot selected TNT screen metrics for selected genotypes")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=False,
        help="Metric names (space or comma separated). If omitted, all available numeric metrics are used.",
    )
    parser.add_argument("--genotypes", nargs="+", required=True, help="Nickname/Genotype/Simplified nickname list")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to pooled summary feather dataset")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output dir")
    parser.add_argument(
        "--control-mode",
        choices=["emptysplit", "tailored", "tnt_pr"],
        default="emptysplit",
        help="Control selection mode (default: emptysplit)",
    )
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations")
    args = parser.parse_args()

    metrics = parse_list_args(args.metrics) if args.metrics else []
    targets = parse_list_args(args.genotypes)

    print(f"📊 Loading dataset: {args.data_path}")
    df = load_dataset(args.data_path)

    if not metrics:
        metrics = auto_detect_metrics(df)
        print(f"📋 --metrics not provided: auto-detected {len(metrics)} available numeric metrics")
        if metrics:
            preview = ", ".join(metrics[:15])
            suffix = " ..." if len(metrics) > 15 else ""
            print(f"   {preview}{suffix}")
    else:
        missing_metrics = [m for m in metrics if m not in df.columns]
        if missing_metrics:
            print(f"⚠️ Metrics not found and skipped: {missing_metrics}")
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("❌ No valid metrics found in dataset")
        return

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for target in targets:
        nickname = resolve_to_nickname(target, df)
        if nickname is None:
            print(f"⚠️ Could not resolve target '{target}' to a unique nickname - skipping")
            continue

        control = get_control_for_nickname(nickname, args.control_mode)
        if control is None:
            print(f"⚠️ Could not determine control for '{nickname}' - skipping")
            continue

        target_df = df[df["Nickname"] == nickname]
        control_df = df[df["Nickname"] == control]

        if target_df.empty or control_df.empty:
            print(f"⚠️ Missing data for target/control pair: {nickname} vs {control} - skipping")
            continue

        brain_region = brain_region_for_nickname(df, nickname)
        display_name = display_name_for_nickname(df, nickname)
        control_region = brain_region_for_nickname(df, control)

        target_color = region_color(brain_region)
        control_color = region_color(control_region)

        out_dir = output_root / sanitize_name(brain_region) / sanitize_name(display_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎯 {nickname} vs {control}")
        print(f"   Brain region: {brain_region}")
        print(f"   Output: {out_dir}")

        for metric in metrics:
            control_values = control_df[metric].dropna().values
            target_values = target_df[metric].dropna().values

            if len(control_values) < 2 or len(target_values) < 2:
                print(
                    f"   ⚠️ {metric}: insufficient data (control n={len(control_values)}, target n={len(target_values)})"
                )
                continue

            result = plot_metric_box_jitter(
                metric=metric,
                control_name=control,
                target_name=display_name,
                control_values=control_values,
                target_values=target_values,
                output_dir=out_dir,
                n_permutations=args.n_permutations,
                control_color=control_color,
                target_color=target_color,
            )
            result.update(
                {
                    "target_input": target,
                    "target_nickname": nickname,
                    "target_display_name": display_name,
                    "target_brain_region": brain_region,
                    "control_nickname": control,
                    "control_brain_region": control_region,
                    "control_mode": args.control_mode,
                }
            )
            all_rows.append(result)
            print(f"   ✅ {metric}: p={format_p_value(result['p_value'])}, d={result['cohens_d']:.3f}")

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_csv = output_root / "selected_metrics_statistics.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n💾 Saved summary statistics: {summary_csv}")
    else:
        print("\n⚠️ No plots generated")


if __name__ == "__main__":
    main()
