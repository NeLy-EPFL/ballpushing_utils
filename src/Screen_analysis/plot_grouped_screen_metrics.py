#!/usr/bin/env python3
"""
Generate grouped panel layouts for selected TNT screen metrics.

For each requested metric, this script creates one figure with one panel per condition:
- first panel: control (default: Empty-Split)
- following panels: requested genotypes in user-specified order

Each panel shows a single boxplot + jittered points. For each non-control panel,
a two-sided permutation test compares target vs control. Significant differences are
annotated with red stars above the target panel, matching the selected screen style.

Examples
--------
python plot_grouped_screen_metrics.py \
  --metrics velocity_during_interactions first_major_event distance_ratio \
  --genotypes LC10-2 TNTxMB247 DDC-gal4

python plot_grouped_screen_metrics.py \
  --metrics velocity_during_interaction,first_major_event,distance_ratio \
  --genotypes LC10-2 TNTxMB247 DDC-Gal4 \
  --panel-width-mm 60 \
  --panel-height-mm 85
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Config


# Illustrator-editable text
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


DEFAULT_DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
DEFAULT_OUTPUT_ROOT = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics/Grouped_layouts"

FONT_SIZE_TICKS = 6
FONT_SIZE_LABELS = 7
FONT_SIZE_ANNOTATIONS = 7
DPI = 300

# Facet geometry tuning
GROUPED_BOX_WIDTH = 0.38
GROUPED_SCATTER_SIZE = 18

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
    "normalized_velocity": "Normalized walking velocity",
    "first_major_event_time": "First major (>1.2mm) event time (s)",
    "max_event_time": "Max ball displ. time (s)",
    "nb_freeze": "short pauses (>2s <5px) (#)",
    "flailing": "Movement of front legs during contact",
    "velocity_during_interactions": "Fly speed during ball contact (mm/s)",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing (>30deg) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "velocity_trend": "Slope linear fit to fly velocity over time",
}

# Friendly aliases for common typo/variants in CLI input.
METRIC_ALIASES = {
    "velocity_during_interaction": "velocity_during_interactions",
}


def mm_to_inches(mm_value):
    return mm_value / 25.4


def sanitize_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def parse_list_args(values):
    items = []
    for value in values:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        items.extend(parts)
    return list(dict.fromkeys(items))


def normalize_token(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def metric_display_name(metric_name):
    metric_key = str(metric_name)
    return METRIC_DISPLAY_NAMES.get(metric_key, metric_key)


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


def ensure_reportable_p_value(p_value):
    """Return finite p-values bounded away from exact zero for CSV/reporting."""
    if p_value is None or pd.isna(p_value):
        return np.nan
    try:
        p_float = float(p_value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(p_float):
        return np.nan
    smallest_positive = float(np.nextafter(0.0, 1.0))
    return float(min(max(p_float, smallest_positive), 1.0))


def benjamini_hochberg_correction(p_values):
    """Compute BH-FDR adjusted p-values while preserving input order."""
    p_vals = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_vals.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p_vals)
    finite_idx = np.where(finite_mask)[0]
    m = len(finite_idx)
    if m == 0:
        return adjusted

    ranked_order = finite_idx[np.argsort(p_vals[finite_idx])]
    ranked_p = p_vals[ranked_order]

    adjusted_ranked = np.empty(m, dtype=float)
    for i in range(m - 1, -1, -1):
        rank = i + 1
        bh_value = ranked_p[i] * m / rank
        if i == m - 1:
            adjusted_ranked[i] = bh_value
        else:
            adjusted_ranked[i] = min(bh_value, adjusted_ranked[i + 1])

    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted[ranked_order] = adjusted_ranked
    return adjusted


def apply_corrections_to_stats_rows(stats_rows, alpha=0.05):
    """Apply BH-FDR correction per metric across target-vs-control comparisons."""
    if not stats_rows:
        return stats_rows

    rows_by_metric = {}
    for idx, row in enumerate(stats_rows):
        rows_by_metric.setdefault(row.get("metric"), []).append(idx)

    for _, row_indices in rows_by_metric.items():
        raw_pvals = [ensure_reportable_p_value(stats_rows[row_idx].get("p_value")) for row_idx in row_indices]
        corrected = benjamini_hochberg_correction(raw_pvals)

        for local_idx, row_idx in enumerate(row_indices):
            p_raw = raw_pvals[local_idx]
            p_corr = corrected[local_idx] if local_idx < len(corrected) else np.nan

            stats_rows[row_idx]["alpha"] = float(alpha)
            stats_rows[row_idx]["test_name"] = "two-sided permutation test on mean difference"
            stats_rows[row_idx]["p_value"] = p_raw
            stats_rows[row_idx]["p_value_corrected"] = ensure_reportable_p_value(p_corr)
            stats_rows[row_idx]["p_value_correction"] = "Benjamini-Hochberg (FDR)"
            stats_rows[row_idx]["n_comparisons_correction"] = int(np.sum(np.isfinite(raw_pvals)))
            stats_rows[row_idx]["significance_raw"] = significance_label(p_raw)
            stats_rows[row_idx]["significance_corrected"] = significance_label(stats_rows[row_idx]["p_value_corrected"])
            stats_rows[row_idx]["is_significant_raw"] = bool(np.isfinite(p_raw) and p_raw < alpha)
            stats_rows[row_idx]["is_significant_corrected"] = bool(
                np.isfinite(stats_rows[row_idx]["p_value_corrected"])
                and stats_rows[row_idx]["p_value_corrected"] < alpha
            )

    return stats_rows


def build_ordered_statistics_dataframe(stats_rows):
    """Build a review-friendly statistics table with a stable column order."""
    stats_df = pd.DataFrame(stats_rows)

    preferred_columns = [
        "analysis_type",
        "method",
        "metric",
        "metric_display_name",
        "comparison",
        "control_nickname",
        "target_nickname",
        "target_display_name",
        "target_region",
        "n_control",
        "n_target",
        "control_mean",
        "target_mean",
        "control_std",
        "target_std",
        "control_median",
        "target_median",
        "mean_diff",
        "median_diff",
        "effect_size_raw",
        "cohens_d",
        "pct_change",
        "ci_lower",
        "ci_upper",
        "pct_ci_lower",
        "pct_ci_upper",
        "bootstrap_n",
        "test_name",
        "p_value",
        "p_value_corrected",
        "p_value_correction",
        "alpha",
        "significance_raw",
        "significance_corrected",
        "is_significant_raw",
        "is_significant_corrected",
        "n_comparisons_correction",
        "n_permutations",
        "plot_pdf",
        "plot_png",
        "plot_svg",
    ]

    ordered_existing = [col for col in preferred_columns if col in stats_df.columns]
    remaining = [col for col in stats_df.columns if col not in ordered_existing]
    return stats_df[ordered_existing + remaining]


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


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrap CI for mean(group2) - mean(group1)."""
    if len(group1) == 0 or len(group2) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    lower = float(np.percentile(bootstrap_diffs, alpha / 2))
    upper = float(np.percentile(bootstrap_diffs, 100 - alpha / 2))
    return lower, upper


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

    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    registry_cols = ["Nickname", "Simplified Nickname", "Simplified region", "Split"]
    reg = Config.SplitRegistry[[c for c in registry_cols if c in Config.SplitRegistry.columns]].copy()
    if "Nickname" in reg.columns:
        reg = reg[~reg["Nickname"].duplicated(keep="first")]

    missing_mapping_cols = [
        "Simplified Nickname" not in df.columns,
        "Simplified region" not in df.columns,
        "Split" not in df.columns,
    ]
    if any(missing_mapping_cols) and "Nickname" in df.columns:
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
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    metrics = [
        col for col in numeric_cols if col not in METADATA_COLUMNS_FOR_METRICS and not col.lower().startswith("unnamed")
    ]
    metrics = [col for col in metrics if col.lower() not in {"frame", "time", "id"}]
    return sorted(metrics)


def resolve_metric_name(metric_input, available_columns):
    metric_raw = str(metric_input).strip()
    if not metric_raw:
        return None

    metric_candidate = METRIC_ALIASES.get(metric_raw, metric_raw)
    if metric_candidate in available_columns:
        return metric_candidate

    lower_matches = [col for col in available_columns if str(col).lower() == metric_candidate.lower()]
    if len(lower_matches) == 1:
        return lower_matches[0]

    metric_norm = normalize_token(metric_candidate)
    norm_matches = [col for col in available_columns if normalize_token(col) == metric_norm]
    if len(norm_matches) == 1:
        return norm_matches[0]

    return None


def resolve_to_nickname(target, df):
    target = str(target).strip()
    if "Nickname" not in df.columns:
        return None

    nicknames = df["Nickname"].dropna().astype(str).unique().tolist()
    if target in nicknames:
        return target

    lowered_map = {}
    for nickname in nicknames:
        lowered_map.setdefault(nickname.lower(), []).append(nickname)
    lowered_matches = lowered_map.get(target.lower(), [])
    if len(lowered_matches) == 1:
        return lowered_matches[0]

    target_norm = normalize_token(target)
    if not target_norm:
        return None

    all_matches = []
    for candidate_col in ["Nickname", "Simplified Nickname", "Genotype"]:
        if candidate_col not in df.columns:
            continue

        if candidate_col == "Nickname":
            subset = df[["Nickname"]].dropna()
            candidate_series = subset["Nickname"].astype(str)
            nickname_series = subset["Nickname"].astype(str)
        else:
            subset = df[[candidate_col, "Nickname"]].dropna()
            candidate_series = subset[candidate_col].astype(str)
            nickname_series = subset["Nickname"].astype(str)

        candidate_norm = candidate_series.map(normalize_token)
        col_matches = nickname_series.loc[candidate_norm == target_norm].tolist()
        all_matches.extend(col_matches)

    unique_matches = list(dict.fromkeys(all_matches))
    if len(unique_matches) == 1:
        return unique_matches[0]

    return None


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


def region_color(region_name):
    return Config.color_dict.get(str(region_name), "#7f7f7f")


def compute_metric_plot_settings(df, metric, upper_clip_percentile=99.0):
    values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"clip_upper": 1.0, "y_lower": 0.0, "y_upper": 1.0}

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

    return {
        "clip_upper": upper_clip,
        "y_lower": float(y_lower),
        "y_upper": float(y_upper),
    }


def draw_panel(ax, values, label, color, jitter_amount, scatter_size, point_alpha, rng):
    if len(values) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_SIZE_TICKS,
            color="#666666",
        )
        ax.set_xticks([])
        return

    bp = ax.boxplot(
        [values],
        positions=[0],
        widths=GROUPED_BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Keep style aligned with the current selected-metric and magnetblock-like style.
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_alpha(1.0)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    x = rng.normal(0.0, jitter_amount, size=len(values))
    ax.scatter(
        x,
        values,
        s=scatter_size,
        alpha=point_alpha,
        c=color,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    ax.set_xticks([0])
    ax.set_xticklabels([f"{label}\n(n={len(values)})"], fontsize=FONT_SIZE_TICKS)


def plot_grouped_metric_layout(
    metric,
    panel_nicknames,
    df,
    output_dir,
    panel_width_mm,
    panel_height_mm,
    n_permutations,
    y_clip_percentile,
    seed,
    n_bootstrap,
):
    control_nickname = panel_nicknames[0]
    metric_settings = compute_metric_plot_settings(df, metric, upper_clip_percentile=y_clip_percentile)
    clip_upper = float(metric_settings["clip_upper"])
    y_lower = float(metric_settings["y_lower"])
    y_upper = float(metric_settings["y_upper"])
    y_range = max(y_upper - y_lower, 1e-9)

    panel_info = []
    for panel_nickname in panel_nicknames:
        region = brain_region_for_nickname(df, panel_nickname)
        display_name = display_name_for_nickname(df, panel_nickname)
        color = region_color(region)
        raw_values = (
            pd.to_numeric(
                df[df["Nickname"] == panel_nickname][metric],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        plot_values = np.clip(raw_values, a_min=None, a_max=clip_upper)

        panel_info.append(
            {
                "nickname": panel_nickname,
                "display_name": display_name,
                "region": region,
                "color": color,
                "raw_values": raw_values,
                "plot_values": plot_values,
            }
        )

    n_panels = len(panel_info)
    fig_width = mm_to_inches(panel_width_mm * n_panels)
    fig_height = mm_to_inches(panel_height_mm)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), sharey=True)
    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(seed)
    stats_rows = []

    control_raw_values = panel_info[0]["raw_values"]

    for idx, (ax, panel) in enumerate(zip(axes, panel_info)):
        is_control = idx == 0
        draw_panel(
            ax=ax,
            values=panel["plot_values"],
            label=panel["display_name"],
            color=panel["color"],
            jitter_amount=0.04,
            scatter_size=GROUPED_SCATTER_SIZE,
            point_alpha=0.48 if is_control else 0.8,
            rng=rng,
        )

        ax.set_ylim(y_lower, y_upper)
        ax.set_facecolor("#f5f5f5" if is_control else "white")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.set_xlim(-0.35, 0.35)
        ax.tick_params(axis="both", which="major", direction="out", length=3, width=1.0, labelsize=FONT_SIZE_TICKS)

        # Facet-like styling: keep only one shared y-axis on the first panel.
        if idx == 0:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_linewidth(1.0)
            ax.tick_params(axis="y", left=True, labelleft=True, labelsize=FONT_SIZE_TICKS)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

        if idx < n_panels - 1:
            # Thin separator to make panel boundaries explicit.
            ax.plot(
                [1.0, 1.0],
                [0.04, 0.96],
                transform=ax.transAxes,
                color="black",
                linewidth=0.8,
                clip_on=False,
            )

        if is_control:
            continue

        target_raw_values = panel["raw_values"]
        if len(control_raw_values) < 2 or len(target_raw_values) < 2:
            p_value = np.nan
            sig = "n/a"
        else:
            p_value = permutation_test_1d(
                control_raw_values,
                target_raw_values,
                n_permutations=n_permutations,
                random_state=42,
            )
            sig = significance_label(p_value)

        if np.isfinite(p_value) and p_value < 0.05:
            sig_y = min(y_upper - 0.03 * y_range, clip_upper + 0.08 * y_range)
            ax.text(
                0.0,
                sig_y,
                sig,
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_ANNOTATIONS,
                fontweight="bold",
                color="red",
            )

        d_value = cohens_d(target_raw_values, control_raw_values)

        control_mean = float(np.mean(control_raw_values)) if len(control_raw_values) > 0 else np.nan
        target_mean = float(np.mean(target_raw_values)) if len(target_raw_values) > 0 else np.nan
        control_std = float(np.std(control_raw_values, ddof=1)) if len(control_raw_values) > 1 else np.nan
        target_std = float(np.std(target_raw_values, ddof=1)) if len(target_raw_values) > 1 else np.nan
        control_median = float(np.median(control_raw_values)) if len(control_raw_values) > 0 else np.nan
        target_median = float(np.median(target_raw_values)) if len(target_raw_values) > 0 else np.nan

        mean_diff = target_mean - control_mean if np.isfinite(control_mean) and np.isfinite(target_mean) else np.nan
        median_diff = (
            target_median - control_median if np.isfinite(control_median) and np.isfinite(target_median) else np.nan
        )

        ci_lower, ci_upper = bootstrap_ci_difference(
            control_raw_values,
            target_raw_values,
            n_bootstrap=n_bootstrap,
            ci=95,
            random_state=seed + idx,
        )

        if np.isfinite(control_mean) and control_mean != 0 and np.isfinite(mean_diff):
            pct_change = (mean_diff / control_mean) * 100.0
            pct_ci_lower = (ci_lower / control_mean) * 100.0 if np.isfinite(ci_lower) else np.nan
            pct_ci_upper = (ci_upper / control_mean) * 100.0 if np.isfinite(ci_upper) else np.nan
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan

        stats_rows.append(
            {
                "analysis_type": "continuous",
                "method": "permutation",
                "metric": metric,
                "metric_display_name": metric_display_name(metric),
                "comparison": "target_vs_control",
                "control_nickname": control_nickname,
                "target_nickname": panel["nickname"],
                "target_display_name": panel["display_name"],
                "target_region": panel["region"],
                "n_control": len(control_raw_values),
                "n_target": len(target_raw_values),
                "control_mean": control_mean,
                "target_mean": target_mean,
                "control_std": control_std,
                "target_std": target_std,
                "control_median": control_median,
                "target_median": target_median,
                "delta_target_minus_control": mean_diff,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "effect_size_raw": mean_diff,
                "pct_change": pct_change,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_ci_lower": pct_ci_lower,
                "pct_ci_upper": pct_ci_upper,
                "bootstrap_n": int(n_bootstrap),
                "cohens_d": float(d_value) if not np.isnan(d_value) else np.nan,
                "p_value": ensure_reportable_p_value(p_value),
                "significance": sig,
                "n_permutations": int(n_permutations),
            }
        )

    axes[0].set_ylabel(metric_display_name(metric), fontsize=FONT_SIZE_LABELS)
    fig.suptitle(metric_display_name(metric), fontsize=FONT_SIZE_LABELS, y=0.98)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.2, wspace=0.08)

    metric_safe = sanitize_name(metric)
    panel_tag = "__".join(sanitize_name(p["display_name"]) for p in panel_info)
    base_name = f"grouped_{metric_safe}__{panel_tag}"

    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"

    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    for row in stats_rows:
        row["plot_pdf"] = str(pdf_path)
        row["plot_png"] = str(png_path)
        row["plot_svg"] = str(svg_path)

    return stats_rows, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate grouped metric layouts for selected TNT screen genotypes")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric names (space or comma separated)",
    )
    parser.add_argument(
        "--genotypes",
        nargs="+",
        required=True,
        help="Ordered genotype list (Nickname/Genotype/Simplified nickname)",
    )
    parser.add_argument(
        "--control",
        default="Empty-Split",
        help="Control nickname/genotype/simplified nickname (default: Empty-Split)",
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to pooled summary feather dataset")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output directory")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations per comparison")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CI and percent-change statistics (default: 10000)",
    )
    parser.add_argument(
        "--y-clip-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for visualization clipping within each metric (default: 99)",
    )
    parser.add_argument(
        "--panel-width-mm",
        type=float,
        default=20.0,
        help="Panel width in mm (default: 20)",
    )
    parser.add_argument(
        "--panel-height-mm",
        type=float,
        default=45.0,
        help="Panel height in mm (default: 45)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data_path}")
    df = load_dataset(args.data_path)

    metrics_input = parse_list_args(args.metrics)
    genotypes_input = parse_list_args(args.genotypes)

    if args.y_clip_percentile <= 0 or args.y_clip_percentile > 100:
        raise ValueError("--y-clip-percentile must be in (0, 100]")
    if args.panel_width_mm <= 0 or args.panel_height_mm <= 0:
        raise ValueError("--panel-width-mm and --panel-height-mm must be positive")
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be a positive integer")

    available_columns = list(df.columns)
    resolved_metrics = []
    for metric_input in metrics_input:
        metric_name = resolve_metric_name(metric_input, available_columns)
        if metric_name is None:
            print(f"WARNING: metric '{metric_input}' not found - skipped")
            continue
        resolved_metrics.append(metric_name)
    resolved_metrics = list(dict.fromkeys(resolved_metrics))

    if not resolved_metrics:
        auto_metrics = auto_detect_metrics(df)
        raise ValueError("No valid metrics resolved from --metrics. " f"Try one of: {', '.join(auto_metrics[:20])}")

    control_nickname = resolve_to_nickname(args.control, df)
    if control_nickname is None:
        raise ValueError(f"Could not resolve control '{args.control}' to a unique nickname")

    resolved_targets = []
    unresolved_targets = []
    for genotype_input in genotypes_input:
        target_nickname = resolve_to_nickname(genotype_input, df)
        if target_nickname is None:
            unresolved_targets.append(genotype_input)
            continue
        if target_nickname == control_nickname:
            print(f"WARNING: target '{genotype_input}' resolves to control '{control_nickname}' and is skipped")
            continue
        resolved_targets.append(target_nickname)

    if unresolved_targets:
        print(f"WARNING: unresolved genotypes skipped: {unresolved_targets}")

    resolved_targets = list(dict.fromkeys(resolved_targets))
    if not resolved_targets:
        raise ValueError("No valid target genotypes resolved from --genotypes")

    panel_nicknames = [control_nickname] + resolved_targets

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  control: {control_nickname}")
    print(f"  targets (ordered): {resolved_targets}")
    print(f"  metrics: {resolved_metrics}")
    print(f"  panel size: {args.panel_width_mm:.1f} x {args.panel_height_mm:.1f} mm")
    print(f"  bootstrap samples for detailed stats: {args.n_bootstrap}")

    all_stats = []
    for metric in resolved_metrics:
        stats_rows, plot_path = plot_grouped_metric_layout(
            metric=metric,
            panel_nicknames=panel_nicknames,
            df=df,
            output_dir=output_dir,
            panel_width_mm=args.panel_width_mm,
            panel_height_mm=args.panel_height_mm,
            n_permutations=args.n_permutations,
            y_clip_percentile=args.y_clip_percentile,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
        )
        all_stats.extend(stats_rows)
        print(f"  saved {metric}: {plot_path}")

    if all_stats:
        all_stats = apply_corrections_to_stats_rows(all_stats, alpha=0.05)

        print("Detailed statistics:")
        for metric in resolved_metrics:
            for row in [r for r in all_stats if r.get("metric") == metric]:
                print(
                    "    "
                    f"{row['target_display_name']} vs {row['control_nickname']}: "
                    f"p={format_p_value(row['p_value'])}, "
                    f"p_corr={format_p_value(row['p_value_corrected'])}, "
                    f"d={row['cohens_d']:.3f}, "
                    f"pct={row['pct_change']:.2f}%"
                )

        stats_df = build_ordered_statistics_dataframe(all_stats)
        stats_path = output_dir / "grouped_metric_layout_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved statistics: {stats_path}")


if __name__ == "__main__":
    main()
