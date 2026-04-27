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

# Keep display labels aligned with plot_detailed_metric_statistics.py
METRIC_DISPLAY_NAMES = {
    "pulling_ratio": "Proportion pull vs push",
    "distance_ratio": "Dist. ball moved / corridor length",
    "distance_moved": "Dist. ball moved",
    "pulled": "Signif. (>0.3 mm) pulling events (#)",
    "max_event": "Event max. ball displ. (n)",
    "number_of_pauses": "Long pauses (>5s <5px) (#)",
    "first_major_event": "First major (>1.2mm) event(n)",
    "significant_ratio": "Fraction signif. (>0.3 mm) events",
    "max_distance": "Max ball displacement (mm)",
    "chamber_ratio": "Fraction time in chamber",
    "nb_events": "Events (< 2mm fly-ball dist.)(#)",
    "persistence_at_end": "Fraction time near end of corridor",
    "time_chamber_beginning": "Time in chamber first 25% exp. (s)",
    "normalized_speed": "Normalized walking speed",
    "first_major_event_time": "First major (>1.2mm) event time (s)",
    "max_event_time": "Max ball displ. time (s)",
    "nb_freeze": "short pauses (>2s <5px) (#)",
    "flailing": "Movement of front legs during contact",
    "speed_during_interactions": "Fly speed during ball contact (mm/s)",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing (>30deg) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "speed_trend": "Slope linear fit to fly speed over time",
}

# Match F1 TNT plot geometry/typography for cross-figure compositing.
PAPER_FIGURE_HEIGHT_MM = 85
PAPER_FIGURE_WIDTH_MM_MIN = 85
PAPER_FIGURE_WIDTH_MM_MAX = 85

FONT_SIZE_TICKS = 6
FONT_SIZE_LABELS = 7
FONT_SIZE_ANNOTATIONS = 7
FONT_SIZE_PVALUE = 6

DPI = 300


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


def metric_display_name(metric_name):
    metric_key = str(metric_name)
    return METRIC_DISPLAY_NAMES.get(metric_key, metric_key)


def mm_to_inches(mm_value):
    return mm_value / 25.4


def get_adaptive_styling_params(n_groups):
    """Mirror the adaptive panel sizing used in F1 TNT plotting."""
    if n_groups <= 3:
        jitter = 0.05
    elif n_groups <= 5:
        jitter = 0.06
    elif n_groups <= 10:
        jitter = 0.07
    else:
        jitter = 0.08

    width_mm = 12 + 9 * n_groups
    width_mm = max(PAPER_FIGURE_WIDTH_MM_MIN, min(PAPER_FIGURE_WIDTH_MM_MAX, width_mm))

    if n_groups <= 2:
        scatter_size = 14
    elif n_groups <= 4:
        scatter_size = 12
    elif n_groups <= 6:
        scatter_size = 10
    else:
        scatter_size = 8

    return {
        "jitter_amount": jitter,
        "figure_size": (mm_to_inches(width_mm), mm_to_inches(PAPER_FIGURE_HEIGHT_MM)),
        "scatter_size": scatter_size,
    }


def compute_shared_metric_plot_settings(df, metrics, upper_clip_percentile=99.0):
    """
    Compute shared plotting limits per metric from the full dataset.

    For each metric:
    - upper clip is the chosen percentile (e.g. 99th)
    - lower bound is the global minimum
    - y-limits are shared across all plots for that metric
    """
    settings = {}
    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue

        lower_data = float(np.nanmin(values))
        upper_clip = float(np.nanpercentile(values, upper_clip_percentile))
        max_data = float(np.nanmax(values))

        if not np.isfinite(upper_clip):
            upper_clip = max_data
        if upper_clip < lower_data:
            upper_clip = max_data

        y_range = max(upper_clip - lower_data, 1e-9)
        y_lower = lower_data - 0.05 * y_range
        y_upper = upper_clip + 0.15 * y_range

        if lower_data >= 0 and y_lower < 0:
            y_lower = 0.0

        settings[metric] = {
            "clip_upper": upper_clip,
            "y_lower": float(y_lower),
            "y_upper": float(y_upper),
        }

    return settings


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
    shared_metric_settings=None,
):
    style = get_adaptive_styling_params(n_groups=2)
    fig, ax = plt.subplots(figsize=style["figure_size"])

    box_data = [control_values, target_values]
    positions = [0.0, 0.65]
    box_width = 0.5

    # Clip only for visualization; statistics are computed on original values.
    control_plot_values = control_values
    target_plot_values = target_values
    if shared_metric_settings is not None:
        clip_upper = float(shared_metric_settings["clip_upper"])
        control_plot_values = np.clip(control_values, a_min=None, a_max=clip_upper)
        target_plot_values = np.clip(target_values, a_min=None, a_max=clip_upper)

    bp = ax.boxplot(
        [control_plot_values, target_plot_values],
        positions=positions,
        widths=box_width * 0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Magnetblock-like style: unfilled boxes with black outlines.
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_alpha(1.0)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    x0 = np.random.normal(positions[0], style["jitter_amount"] * box_width, size=len(control_plot_values))
    x1 = np.random.normal(positions[1], style["jitter_amount"] * box_width, size=len(target_plot_values))
    ax.scatter(
        x0,
        control_plot_values,
        s=style["scatter_size"],
        alpha=0.48,
        c=control_color,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        x1,
        target_plot_values,
        s=style["scatter_size"],
        alpha=0.8,
        c=target_color,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    p_value = permutation_test_1d(control_values, target_values, n_permutations=n_permutations, random_state=42)
    sig = significance_label(p_value)

    if shared_metric_settings is not None:
        y_lower = float(shared_metric_settings["y_lower"])
        y_upper = float(shared_metric_settings["y_upper"])
        y_range = max(y_upper - y_lower, 1e-9)
        y_max_for_annotation = float(shared_metric_settings["clip_upper"])
    else:
        y_max_for_annotation = max(np.max(control_plot_values), np.max(target_plot_values))
        y_min_for_limits = min(np.min(control_plot_values), np.min(target_plot_values))
        y_range = max(y_max_for_annotation - y_min_for_limits, 1e-9)
        y_lower = y_min_for_limits - 0.05 * y_range
        y_upper = y_max_for_annotation + 0.15 * y_range
        if y_min_for_limits >= 0 and y_lower < 0:
            y_lower = 0
    ax.set_ylim(y_lower, y_upper)

    # Significant-only annotation, no bar and no p-value textbox.
    if p_value < 0.05:
        sig_y = y_max_for_annotation + 0.08 * y_range
        ax.text(
            float(np.mean(positions)),
            sig_y,
            sig,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOTATIONS,
            fontweight="bold",
            color="red",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{control_name}\n(n={len(control_values)})", f"{target_name}\n(n={len(target_values)})"],
        fontsize=FONT_SIZE_TICKS,
    )
    ax.set_ylabel(metric_display_name(metric), fontsize=FONT_SIZE_LABELS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=1.0, labelsize=FONT_SIZE_TICKS)

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
    svg_path = output_dir / f"{metric_safe}.svg"
    plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    d_value = cohens_d(target_values, control_values)
    return {
        "metric": metric,
        "metric_display_name": metric_display_name(metric),
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
        "plot_svg": str(svg_path),
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
    parser.add_argument(
        "--y-clip-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for shared y-axis clipping per metric (default: 99)",
    )
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

    shared_metric_settings = compute_shared_metric_plot_settings(
        df,
        metrics,
        upper_clip_percentile=args.y_clip_percentile,
    )

    if shared_metric_settings:
        print(
            f"📏 Shared y-axis enabled using full dataset with upper clipping at {args.y_clip_percentile:.1f}th percentile"
        )

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
                shared_metric_settings=shared_metric_settings.get(metric),
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
