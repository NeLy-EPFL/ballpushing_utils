#!/usr/bin/env python3
"""
F1 TNT Training-vs-Test Ball Genotype Comparison

What this script does (pretrained flies only):
1) Training-ball jitter boxplots with permutation stats (TNTxEmptySplit vs each genotype)
2) Test-ball jitter boxplots with permutation stats (TNTxEmptySplit vs each genotype)
3) Training-vs-test scatterplots per metric (paired by fly)
4) CSV outputs with permutation stats and training->test rank correlation
   (control baseline + genotype comparisons)

Usage examples:
    python f1_tnt_training_test_comparison.py
    python f1_tnt_training_test_comparison.py --show
    python f1_tnt_training_test_comparison.py --metrics distance_moved,max_distance,nb_events
    python f1_tnt_training_test_comparison.py --dataset /path/to/pooled_summary.feather
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
DATASET_PATH = (
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260304_11_F1_coordinates_F1_TNT_Full_Data/summary/"
    "pooled_summary.feather"
)

# Target pooled path (once available)
POOLED_DATASET_PATH = (
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260304_11_F1_coordinates_F1_TNT_Full_Data/summary/"
    "pooled_summary.feather"
)

OUTPUT_DIR = "/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/training_vs_test_comparison"

# ---------------------------------------------------------------------------
# Styling (aligned with existing TNT genotype script)
# ---------------------------------------------------------------------------
PIXELS_PER_MM = 500 / 30

FONT_SIZE_TICKS = 6
FONT_SIZE_LABELS = 7
FONT_SIZE_TITLE = 8
FONT_SIZE_LEGEND = 6
FONT_SIZE_N_TEXT = 5

DPI = 300
PLOT_FORMATS = ["png", "pdf", "svg"]
SCATTER_ALPHA = 0.6

PAPER_FIGURE_HEIGHT_MM = 25 * 5
PAPER_FIGURE_WIDTH_MM_MIN = 25 * 5
PAPER_FIGURE_WIDTH_MM_MAX = 125 * 5

PREFERRED_CONTROL_GENOTYPE = "TNTxEmptySplit"
FALLBACK_CONTROL_GENOTYPE = "TNTxEmptyGal4"

PREFERRED_SELECTED_GENOTYPES = [
    "TNTxEmptySplit",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

GENOTYPE_COLORS = {
    "TNTxEmptyGal4": "#7f7f7f",
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxPR": "#7f7f7f",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "PRxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "PRxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
    "PRxLC16-1": "#ff7f0e",
}

EXCLUDED_COLUMNS = {
    "fly",
    "Date",
    "date",
    "filename",
    "video",
    "path",
    "folder",
    "Genotype",
    "genotype",
    "Pretraining",
    "pretraining",
    "ball_condition",
    "ball_identity",
    "arena",
    "corridor",
    "time",
    "frame",
    "index",
}

BINARY_METRICS = {"has_major", "has_finished", "has_significant", "has_long_pauses"}

DEFAULT_CONTINUOUS_METRICS = [
    "ball_proximity_proportion_70px",
    "ball_proximity_proportion_140px",
    "ball_proximity_proportion_200px",
    "distance_moved",
    "max_distance",
    "nb_events",
    "significant_ratio",
    "raw_distance_moved",
    "raw_max_distance",
    "time_to_first_interaction",
    "max_event_time",
    "final_event_time",
    "first_significant_event_time",
    "chamber_time",
]

# Bonus/supplementary continuous metrics (mirrors TNT genotype supplementary set where applicable)
SUPPLEMENTARY_CONTINUOUS_METRICS = [
    "max_event",
    "max_event_time",
    "final_event",
    "final_event_time",
    "distance_ratio",
    "chamber_time",
    "chamber_ratio",
    "chamber_exit_time",
    "nb_significant_events",
    "time_to_first_interaction",
    "first_significant_event",
    "first_significant_event_time",
    "first_major_event",
    "first_major_event_time",
    "pulled",
    "pushed",
    "pulling_ratio",
    "interaction_persistence",
    "normalized_velocity",
    "velocity_during_interactions",
    "nb_long_pauses",
    "median_long_pause_duration",
    "total_pause_duration",
    "pauses_persistence",
    "nb_freeze",
    "median_freeze_duration",
    "interaction_proportion",
    "cumulated_breaks_duration",
    "fly_distance_moved",
    "persistence_at_end",
    "time_chamber_beginning",
    "fraction_not_facing_ball",
    "head_pushing_ratio",
    "leg_visibility_ratio",
    "flailing",
]

DEFAULT_METRICS_WITH_SUPPLEMENTARY = list(dict.fromkeys(DEFAULT_CONTINUOUS_METRICS + SUPPLEMENTARY_CONTINUOUS_METRICS))
N_BOOTSTRAP = 2000

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def mm_to_inches(mm_value):
    return mm_value / 25.4


def get_adaptive_styling_params(n_genotypes):
    if n_genotypes <= 3:
        jitter = 0.05
    elif n_genotypes <= 5:
        jitter = 0.06
    elif n_genotypes <= 10:
        jitter = 0.07
    else:
        jitter = 0.08

    width_mm = 12 + 9 * n_genotypes
    width_mm = max(PAPER_FIGURE_WIDTH_MM_MIN, min(PAPER_FIGURE_WIDTH_MM_MAX, width_mm))
    fig_width = mm_to_inches(width_mm)
    fig_height = mm_to_inches(PAPER_FIGURE_HEIGHT_MM)

    if n_genotypes <= 2:
        scatter_size = 14
    elif n_genotypes <= 4:
        scatter_size = 12
    elif n_genotypes <= 6:
        scatter_size = 10
    else:
        scatter_size = 8

    return {
        "jitter_amount": jitter,
        "figure_size": (fig_width, fig_height),
        "scatter_size": scatter_size,
    }


def is_distance_metric(metric_name):
    distance_keywords = ["distance", "dist", "head_ball", "fly_distance_moved", "max_distance", "distance_moved"]
    return any(keyword in metric_name.lower() for keyword in distance_keywords)


def is_time_metric(metric_name):
    time_keywords = ["time", "duration", "pause", "stop", "freeze", "chamber_exit_time", "time_chamber_beginning"]
    if "ratio" in metric_name.lower():
        return False
    return any(keyword in metric_name.lower() for keyword in time_keywords)


def convert_metric_data(data, metric_name):
    if is_distance_metric(metric_name):
        return data / PIXELS_PER_MM
    if is_time_metric(metric_name):
        return data / 60
    return data


def get_metric_unit(metric_name):
    if is_distance_metric(metric_name):
        return "mm"
    if is_time_metric(metric_name):
        return "min"
    return ""


def get_elegant_metric_name(metric_name):
    metric_name_map = {
        "nb_events": "Number of events",
        "max_event_time": "Max event time",
        "final_event_time": "Final event time",
        "first_significant_event_time": "First significant event time",
        "distance_moved": "Total ball distance moved",
        "max_distance": "Max ball distance",
        "raw_distance_moved": "Raw distance moved",
        "raw_max_distance": "Raw max distance",
        "significant_ratio": "Significant event ratio",
        "time_to_first_interaction": "Time to first interaction",
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "ball_proximity_proportion_70px": "Ball proximity (<70px)",
        "ball_proximity_proportion_140px": "Ball proximity (<140px)",
        "ball_proximity_proportion_200px": "Ball proximity (<200px)",
    }
    return metric_name_map.get(metric_name, metric_name.replace("_", " ").title())


def format_metric_label(metric_name):
    elegant_name = get_elegant_metric_name(metric_name)
    unit = get_metric_unit(metric_name)
    return f"{elegant_name} ({unit})" if unit else elegant_name


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def ensure_reportable_p_value(p_value, min_value=None):
    """Ensure exported p-values are finite and never exactly zero."""
    if pd.isna(p_value):
        return np.nan

    try:
        p_float = float(p_value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(p_float):
        return np.nan

    smallest_positive = float(np.nextafter(0.0, 1.0))
    floor = smallest_positive if min_value is None else max(float(min_value), smallest_positive)
    return float(min(max(p_float, floor), 1.0))


def bootstrap_ci_difference(control, genotype, n_bootstrap=2000, confidence=0.95, random_seed=42):
    """Bootstrap CIs for genotype-control difference on raw and percent scales."""
    control_arr = np.asarray(control, dtype=float)
    genotype_arr = np.asarray(genotype, dtype=float)
    control_arr = control_arr[np.isfinite(control_arr)]
    genotype_arr = genotype_arr[np.isfinite(genotype_arr)]

    if len(control_arr) < 2 or len(genotype_arr) < 2 or n_bootstrap < 1:
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    raw_boot = np.empty(n_bootstrap, dtype=float)
    pct_boot = np.full(n_bootstrap, np.nan, dtype=float)

    for i in range(n_bootstrap):
        control_sample = rng.choice(control_arr, size=len(control_arr), replace=True)
        genotype_sample = rng.choice(genotype_arr, size=len(genotype_arr), replace=True)

        mean_control = float(np.mean(control_sample))
        mean_genotype = float(np.mean(genotype_sample))
        diff = mean_genotype - mean_control
        raw_boot[i] = diff

        if mean_control != 0:
            pct_boot[i] = (diff / mean_control) * 100.0

    alpha = 1.0 - confidence
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    raw_ci_bounds = np.asarray(np.percentile(raw_boot, [lower_q, upper_q]), dtype=float).reshape(-1)
    raw_ci_lower = float(raw_ci_bounds[0]) if raw_ci_bounds.size > 0 else np.nan
    raw_ci_upper = float(raw_ci_bounds[1]) if raw_ci_bounds.size > 1 else np.nan

    pct_valid = pct_boot[np.isfinite(pct_boot)]
    if len(pct_valid) > 0:
        pct_ci_bounds = np.asarray(np.percentile(pct_valid, [lower_q, upper_q]), dtype=float).reshape(-1)
        pct_ci_lower = float(pct_ci_bounds[0]) if pct_ci_bounds.size > 0 else np.nan
        pct_ci_upper = float(pct_ci_bounds[1]) if pct_ci_bounds.size > 1 else np.nan
    else:
        pct_ci_lower, pct_ci_upper = np.nan, np.nan

    return (
        raw_ci_lower,
        raw_ci_upper,
        float(pct_ci_lower) if np.isfinite(pct_ci_lower) else np.nan,
        float(pct_ci_upper) if np.isfinite(pct_ci_upper) else np.nan,
    )


def compute_detailed_group_statistics(control, genotype, n_bootstrap=2000):
    """Compute mean/median/raw/percent effects and bootstrap CIs for CSV exports."""
    control_arr = np.asarray(control, dtype=float)
    genotype_arr = np.asarray(genotype, dtype=float)
    control_arr = control_arr[np.isfinite(control_arr)]
    genotype_arr = genotype_arr[np.isfinite(genotype_arr)]

    if len(control_arr) == 0 or len(genotype_arr) == 0:
        return {
            "median_control": np.nan,
            "median_genotype": np.nan,
            "raw_difference": np.nan,
            "percent_change_vs_control": np.nan,
            "raw_ci_lower": np.nan,
            "raw_ci_upper": np.nan,
            "percent_ci_lower": np.nan,
            "percent_ci_upper": np.nan,
            "bootstrap_n": int(max(0, n_bootstrap)),
        }

    mean_control = float(np.mean(control_arr))
    mean_genotype = float(np.mean(genotype_arr))
    raw_difference = float(mean_genotype - mean_control)
    pct_change = float((raw_difference / mean_control) * 100.0) if mean_control != 0 else np.nan

    raw_ci_lower, raw_ci_upper, percent_ci_lower, percent_ci_upper = bootstrap_ci_difference(
        control_arr,
        genotype_arr,
        n_bootstrap=n_bootstrap,
    )

    return {
        "median_control": float(np.median(control_arr)),
        "median_genotype": float(np.median(genotype_arr)),
        "raw_difference": raw_difference,
        "percent_change_vs_control": pct_change,
        "raw_ci_lower": raw_ci_lower,
        "raw_ci_upper": raw_ci_upper,
        "percent_ci_lower": percent_ci_lower,
        "percent_ci_upper": percent_ci_upper,
        "bootstrap_n": int(max(0, n_bootstrap)),
    }


def permutation_test_mean_difference(x, y, n_permutations=10000, rng=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    observed = np.mean(y) - np.mean(x)
    pooled = np.concatenate([x, y])
    n_x = len(x)

    if rng is None:
        rng = np.random.default_rng(42)

    null_dist = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(pooled)
        null_dist[i] = np.mean(perm[n_x:]) - np.mean(perm[:n_x])

    p_value = (np.sum(np.abs(null_dist) >= np.abs(observed)) + 1) / (n_permutations + 1)
    p_value = ensure_reportable_p_value(p_value, min_value=1.0 / (n_permutations + 1))
    return observed, p_value


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / pooled_sd


def to_percentile_ranks(values):
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def safe_spearman_rank_correlation(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]

    if len(x_arr) < 3:
        return None
    if np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return None

    result = spearmanr(x_arr, y_arr)
    result_arr = np.asarray(result, dtype=float).ravel()
    if result_arr.size < 2:
        return None

    rho = float(result_arr[0])
    p_value = float(result_arr[1])

    if not np.isfinite(rho):
        return None

    return {
        "n": int(len(x_arr)),
        "rho": float(rho),
        "rho2": float(rho**2),
        "p_value": ensure_reportable_p_value(p_value),
    }


def compare_correlations_fisher_z(r_control, n_control, r_group, n_group):
    if not np.isfinite(r_control) or not np.isfinite(r_group):
        return np.nan, np.nan
    if n_control < 4 or n_group < 4:
        return np.nan, np.nan

    r1 = float(np.clip(r_control, -0.999999, 0.999999))
    r2 = float(np.clip(r_group, -0.999999, 0.999999))

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1.0 / (n_control - 3) + 1.0 / (n_group - 3))
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan

    z = (z2 - z1) / se
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), ensure_reportable_p_value(p_value)


def compute_group_rank_correlation(group_df, metric):
    training = convert_metric_data(group_df["training"].values, metric)
    test = convert_metric_data(group_df["test"].values, metric)
    training_rank = to_percentile_ranks(training)
    test_rank = to_percentile_ranks(test)
    return safe_spearman_rank_correlation(training_rank, test_rank)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def detect_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def resolve_active_genotypes(df, genotype_col):
    available = [g for g in sorted(df[genotype_col].dropna().astype(str).unique())]

    if PREFERRED_CONTROL_GENOTYPE in available:
        control_genotype = PREFERRED_CONTROL_GENOTYPE
    elif FALLBACK_CONTROL_GENOTYPE in available:
        control_genotype = FALLBACK_CONTROL_GENOTYPE
    else:
        control_genotype = available[0] if available else None

    preferred_in_data = [g for g in PREFERRED_SELECTED_GENOTYPES if g in available]
    if preferred_in_data:
        active_genotypes = preferred_in_data.copy()
        if control_genotype and control_genotype not in active_genotypes:
            active_genotypes = [control_genotype] + active_genotypes
    else:
        active_genotypes = available

    return active_genotypes, control_genotype


def filter_pretrained_testable(df, genotype_col, pretraining_col, identity_col, active_genotypes):
    df2 = df.copy()

    # pretrained only
    is_pretrained = df2[pretraining_col].astype(str).str.lower().isin(["y", "yes", "true", "1"])
    df2 = df2[is_pretrained].copy()

    # selected/active genotypes only
    df2 = df2[df2[genotype_col].isin(active_genotypes)].copy()

    # keep training/test only
    id_norm = df2[identity_col].astype(str).str.lower()
    df2 = df2[id_norm.isin(["training", "test"])].copy()

    return df2


def auto_detect_metrics(df):
    metrics = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS or col in BINARY_METRICS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            continue
        if any(keyword in col.lower() for keyword in ["id", "index", "frame"]):
            continue
        metrics.append(col)
    return sorted(metrics)


def build_paired_dataframe(df, metric, fly_col, genotype_col, identity_col):
    mini = df[[fly_col, genotype_col, identity_col, metric]].copy()
    mini = mini.dropna(subset=[metric])

    pivoted = mini.pivot_table(index=[fly_col, genotype_col], columns=identity_col, values=metric, aggfunc="mean")
    if "training" not in pivoted.columns or "test" not in pivoted.columns:
        return pd.DataFrame(columns=[fly_col, genotype_col, "training", "test"])

    pivoted = pivoted.reset_index()
    pivoted = pivoted.dropna(subset=["training", "test"])
    return pivoted.rename(columns={"training": "training", "test": "test"})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_permutation_stats(
    df,
    metric,
    genotype_col,
    identity_col,
    subset_identity,
    n_permutations,
    n_bootstrap,
    active_genotypes,
    control_genotype,
):
    rows = []
    rng = np.random.default_rng(42)

    subset = df[df[identity_col].astype(str).str.lower() == subset_identity].copy()
    if control_genotype is None:
        return pd.DataFrame()

    control = subset[subset[genotype_col] == control_genotype][metric].dropna().values

    for genotype in active_genotypes:
        if genotype == control_genotype:
            continue

        group = subset[subset[genotype_col] == genotype][metric].dropna().values
        effect, p = permutation_test_mean_difference(control, group, n_permutations=n_permutations, rng=rng)
        d = cohen_d(control, group)
        detailed_stats = compute_detailed_group_statistics(control, group, n_bootstrap=n_bootstrap)

        rows.append(
            {
                "metric": metric,
                "subset": subset_identity,
                "control_genotype": control_genotype,
                "genotype": genotype,
                "n_control": int(len(control)),
                "n_genotype": int(len(group)),
                "mean_control": float(np.nanmean(control)) if len(control) else np.nan,
                "mean_genotype": float(np.nanmean(group)) if len(group) else np.nan,
                "effect_mean_diff_genotype_minus_control": effect,
                "cohen_d": d,
                "p_value": ensure_reportable_p_value(p, min_value=1.0 / (n_permutations + 1)),
                "p_value_floor": float(1.0 / (n_permutations + 1)),
                **detailed_stats,
            }
        )

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        valid = stats_df["p_value"].notna()
        stats_df["p_fdr"] = np.nan
        if valid.any():
            _, p_corr, _, _ = multipletests(stats_df.loc[valid, "p_value"], method="fdr_bh")
            stats_df.loc[valid, "p_fdr"] = [ensure_reportable_p_value(x) for x in p_corr]
        stats_df["significance"] = stats_df["p_fdr"].apply(lambda x: significance_stars(x) if pd.notna(x) else "na")

    return stats_df


def compute_training_test_correlations(
    df,
    metrics,
    fly_col,
    genotype_col,
    identity_col,
    active_genotypes,
    control_genotype,
):
    rows = []

    for metric in metrics:
        paired = build_paired_dataframe(df, metric, fly_col, genotype_col, identity_col)
        if paired.empty or control_genotype is None:
            rows.append(
                {
                    "metric": metric,
                    "group": "control_baseline",
                    "group_genotype": control_genotype,
                    "correlation_method": "spearman_rank_percentile",
                    "n": 0,
                    "r": np.nan,
                    "r2": np.nan,
                    "rho": np.nan,
                    "rho2": np.nan,
                    "p_value": np.nan,
                    "delta_rho_vs_control": np.nan,
                    "fisher_z_vs_control": np.nan,
                    "fisher_p_vs_control": np.nan,
                }
            )
            continue

        # Control baseline
        control_paired = paired[paired[genotype_col] == control_genotype]
        control_corr = compute_group_rank_correlation(control_paired, metric) if len(control_paired) >= 3 else None

        if control_corr is not None:
            rows.append(
                {
                    "metric": metric,
                    "group": "control_baseline",
                    "group_genotype": control_genotype,
                    "correlation_method": "spearman_rank_percentile",
                    "n": int(control_corr["n"]),
                    "r": float(control_corr["rho"]),
                    "r2": float(control_corr["rho2"]),
                    "rho": float(control_corr["rho"]),
                    "rho2": float(control_corr["rho2"]),
                    "p_value": float(control_corr["p_value"]),
                    "delta_rho_vs_control": 0.0,
                    "fisher_z_vs_control": np.nan,
                    "fisher_p_vs_control": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "metric": metric,
                    "group": "control_baseline",
                    "group_genotype": control_genotype,
                    "correlation_method": "spearman_rank_percentile",
                    "n": int(len(control_paired)),
                    "r": np.nan,
                    "r2": np.nan,
                    "rho": np.nan,
                    "rho2": np.nan,
                    "p_value": np.nan,
                    "delta_rho_vs_control": np.nan,
                    "fisher_z_vs_control": np.nan,
                    "fisher_p_vs_control": np.nan,
                }
            )

        # Per genotype, compared to control
        for genotype in active_genotypes:
            if genotype == control_genotype:
                continue

            g = paired[paired[genotype_col] == genotype]
            g_corr = compute_group_rank_correlation(g, metric) if len(g) >= 3 else None

            if g_corr is not None:
                if control_corr is not None:
                    fisher_z, fisher_p = compare_correlations_fisher_z(
                        control_corr["rho"], control_corr["n"], g_corr["rho"], g_corr["n"]
                    )
                    delta_rho = float(g_corr["rho"] - control_corr["rho"])
                else:
                    fisher_z, fisher_p = np.nan, np.nan
                    delta_rho = np.nan

                rows.append(
                    {
                        "metric": metric,
                        "group": "genotype",
                        "group_genotype": genotype,
                        "correlation_method": "spearman_rank_percentile",
                        "n": int(g_corr["n"]),
                        "r": float(g_corr["rho"]),
                        "r2": float(g_corr["rho2"]),
                        "rho": float(g_corr["rho"]),
                        "rho2": float(g_corr["rho2"]),
                        "p_value": float(g_corr["p_value"]),
                        "delta_rho_vs_control": delta_rho,
                        "fisher_z_vs_control": fisher_z,
                        "fisher_p_vs_control": fisher_p,
                    }
                )
            else:
                rows.append(
                    {
                        "metric": metric,
                        "group": "genotype",
                        "group_genotype": genotype,
                        "correlation_method": "spearman_rank_percentile",
                        "n": int(len(g)),
                        "r": np.nan,
                        "r2": np.nan,
                        "rho": np.nan,
                        "rho2": np.nan,
                        "p_value": np.nan,
                        "delta_rho_vs_control": np.nan,
                        "fisher_z_vs_control": np.nan,
                        "fisher_p_vs_control": np.nan,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_jitter_box(
    df,
    metric,
    genotype_col,
    identity_col,
    subset_identity,
    stats_df,
    output_dir,
    active_genotypes,
    control_genotype,
    show=False,
):
    subset = df[df[identity_col].astype(str).str.lower() == subset_identity].copy()
    subset = subset[subset[genotype_col].isin(active_genotypes)]

    adaptive = get_adaptive_styling_params(len(active_genotypes))
    fig, ax = plt.subplots(figsize=adaptive["figure_size"])

    data_list = []
    positions = []
    colors = []

    for i, genotype in enumerate(active_genotypes):
        vals = subset.loc[subset[genotype_col] == genotype, metric].dropna().values
        vals = convert_metric_data(vals, metric)
        data_list.append(vals)
        positions.append(i)
        colors.append(GENOTYPE_COLORS.get(genotype, "#808080"))

    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")

    for vals, pos, c in zip(data_list, positions, colors):
        if len(vals) == 0:
            continue
        jitter = np.random.uniform(-adaptive["jitter_amount"], adaptive["jitter_amount"], len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=adaptive["scatter_size"], alpha=SCATTER_ALPHA, c=c)

    labels = []
    for genotype, vals in zip(active_genotypes, data_list):
        labels.append(f"{genotype}\n(n={len(vals)})")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICKS)
    metric_label = str(format_metric_label(metric))
    ax.set_ylabel(metric_label, fontsize=FONT_SIZE_LABELS)
    ax.set_xlabel("Genotype", fontsize=FONT_SIZE_LABELS)
    ax.set_title(f"{get_elegant_metric_name(metric)} - {subset_identity.capitalize()} ball", fontsize=FONT_SIZE_TITLE)

    if stats_df is not None and not stats_df.empty:
        y_max = (
            np.nanmax(np.concatenate([x for x in data_list if len(x) > 0])) if any(len(x) > 0 for x in data_list) else 1
        )
        y_min = (
            np.nanmin(np.concatenate([x for x in data_list if len(x) > 0])) if any(len(x) > 0 for x in data_list) else 0
        )
        y_range = max(1e-6, y_max - y_min)
        ctrl_pos = active_genotypes.index(control_genotype)

        annotation_entries = []
        for i, genotype in enumerate(active_genotypes):
            if genotype == control_genotype:
                continue
            row = stats_df[stats_df["genotype"] == genotype]
            if row.empty:
                continue
            star = row.iloc[0]["significance"]
            annotation_entries.append((i, star))

        annotation_entries.sort(key=lambda item: abs(item[0] - ctrl_pos))
        base_offset = 0.10
        step_offset = 0.08
        cap = 0.015 * y_range

        for level, (i, star) in enumerate(annotation_entries):
            y_anno = y_max + (base_offset + step_offset * level) * y_range
            ax.plot(
                [ctrl_pos, ctrl_pos, i, i],
                [y_anno - cap, y_anno, y_anno, y_anno - cap],
                color="black",
                linewidth=1,
            )
            ax.text(
                (ctrl_pos + i) / 2,
                y_anno + 0.01 * y_range,
                f"{star}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_N_TEXT,
            )

        if annotation_entries:
            top_y = y_max + (base_offset + step_offset * len(annotation_entries) + 0.08) * y_range
            ax.set_ylim(top=top_y)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=1.0, labelsize=FONT_SIZE_TICKS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in PLOT_FORMATS:
        out = output_dir / f"{metric}_{subset_identity}_jitterbox.{fmt}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", format=fmt)

    if show:
        plt.show()
    plt.close(fig)


def plot_training_vs_test_scatter(
    df,
    metric,
    fly_col,
    genotype_col,
    identity_col,
    output_dir,
    active_genotypes,
    control_genotype,
    show=False,
):
    paired = build_paired_dataframe(df, metric, fly_col, genotype_col, identity_col)
    if paired.empty:
        return

    adaptive = get_adaptive_styling_params(len(active_genotypes))
    fig, ax = plt.subplots(figsize=adaptive["figure_size"])

    control_corr = None
    genotype_corr_rows = []

    for genotype in active_genotypes:
        g = paired[paired[genotype_col] == genotype].copy()
        if g.empty:
            continue

        g["training_conv"] = convert_metric_data(g["training"].values, metric)
        g["test_conv"] = convert_metric_data(g["test"].values, metric)
        g["training_rank"] = to_percentile_ranks(g["training_conv"].values)
        g["test_rank"] = to_percentile_ranks(g["test_conv"].values)

        corr = safe_spearman_rank_correlation(g["training_rank"].values, g["test_rank"].values)
        genotype_corr_rows.append((genotype, corr))
        if genotype == control_genotype:
            control_corr = corr

        ax.scatter(
            g["training_rank"],
            g["test_rank"],
            s=adaptive["scatter_size"] + 6,
            alpha=SCATTER_ALPHA,
            c=GENOTYPE_COLORS.get(genotype, "#808080"),
            edgecolors="black",
            linewidths=0.4,
            label=f"{genotype} (n={len(g)})",
        )

    # identity line in percentile-rank space
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1)

    # control trend line (visual aid)
    control_points = paired[paired[genotype_col] == control_genotype].copy()
    if not control_points.empty:
        control_points["training_conv"] = convert_metric_data(control_points["training"].values, metric)
        control_points["test_conv"] = convert_metric_data(control_points["test"].values, metric)
        control_points["training_rank"] = to_percentile_ranks(control_points["training_conv"].values)
        control_points["test_rank"] = to_percentile_ranks(control_points["test_conv"].values)
        if len(control_points) >= 3 and np.nanstd(control_points["training_rank"]) > 0:
            slope, intercept = np.polyfit(control_points["training_rank"], control_points["test_rank"], deg=1)
            x = np.linspace(0, 1, 100)
            y = np.clip(slope * x + intercept, 0, 1)
            ax.plot(x, y, color="black", linestyle="--", linewidth=1.2)

    annotation_lines = ["Rank-aware correlation (within genotype)"]
    if control_corr is not None:
        annotation_lines.append(
            f"{control_genotype}: rho={control_corr['rho']:.2f}, p={control_corr['p_value']:.3g}, n={control_corr['n']}"
        )

    for genotype, corr in genotype_corr_rows:
        if genotype == control_genotype:
            continue
        if corr is None:
            annotation_lines.append(f"{genotype}: rho=na")
            continue

        if control_corr is not None:
            _, p_diff = compare_correlations_fisher_z(control_corr["rho"], control_corr["n"], corr["rho"], corr["n"])
            delta = corr["rho"] - control_corr["rho"]
            p_diff_text = f"{p_diff:.3g}" if np.isfinite(p_diff) else "na"
            annotation_lines.append(f"{genotype}: rho={corr['rho']:.2f}, delta={delta:+.2f}, p(delta)={p_diff_text}")
        else:
            annotation_lines.append(f"{genotype}: rho={corr['rho']:.2f}, p={corr['p_value']:.3g}")

    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE_N_TEXT,
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    label = format_metric_label(metric)
    ax.set_xlabel(f"Training rank (within genotype) - {label}", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel(f"Test rank (within genotype) - {label}", fontsize=FONT_SIZE_LABELS)
    ax.set_title(f"Training vs Test (rank-aware) - {get_elegant_metric_name(metric)}", fontsize=FONT_SIZE_TITLE)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=1.0, labelsize=FONT_SIZE_TICKS)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.22), borderaxespad=0.0, fontsize=FONT_SIZE_LEGEND, frameon=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in PLOT_FORMATS:
        out = output_dir / f"{metric}_training_vs_test_scatter.{fmt}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", format=fmt)

    if show:
        plt.show()
    plt.close(fig)


def plot_training_test_cohens_d_heatmaps(
    perm_stats,
    all_metrics,
    active_genotypes,
    control_genotype,
    output_dir,
    show=False,
):
    if perm_stats is None or perm_stats.empty:
        return

    tested_genotypes = [g for g in active_genotypes if g != control_genotype]
    if not tested_genotypes:
        return

    available_metric_set = set(perm_stats["metric"].astype(str).unique().tolist())
    metric_order = [m for m in all_metrics if m in available_metric_set]
    if not metric_order:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_matrices = {}
    subset_pvals = {}
    for subset_name in ["training", "test"]:
        subset_df = perm_stats[perm_stats["subset"] == subset_name].copy()
        d_matrix = subset_df.pivot_table(index="genotype", columns="metric", values="cohen_d", aggfunc="mean")
        p_matrix = subset_df.pivot_table(index="genotype", columns="metric", values="p_fdr", aggfunc="mean")
        d_matrix = d_matrix.reindex(index=tested_genotypes, columns=metric_order)
        p_matrix = p_matrix.reindex(index=tested_genotypes, columns=metric_order)
        subset_matrices[subset_name] = d_matrix
        subset_pvals[subset_name] = p_matrix

        d_matrix.to_csv(output_dir / f"cohens_d_heatmap_{subset_name}.csv")
        p_matrix.to_csv(output_dir / f"cohens_d_heatmap_{subset_name}_p_fdr.csv")

    combined_vals = np.concatenate(
        [
            subset_matrices["training"].to_numpy(dtype=float).flatten(),
            subset_matrices["test"].to_numpy(dtype=float).flatten(),
        ]
    )
    finite_vals = combined_vals[np.isfinite(combined_vals)]
    if len(finite_vals) == 0:
        return

    vmax = max(1.0, float(np.nanpercentile(np.abs(finite_vals), 95)))

    fig_width = max(12, 0.33 * len(metric_order) * 2)
    fig_height = max(4.5, 0.55 * len(tested_genotypes) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    annotate_values = len(metric_order) <= 24 and len(tested_genotypes) <= 12
    display_metric_labels = [get_elegant_metric_name(m) for m in metric_order]
    im = None

    for ax, subset_name in zip(axes, ["training", "test"]):
        matrix = subset_matrices[subset_name]
        pmat = subset_pvals[subset_name]
        data = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(masked, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{subset_name.capitalize()} (vs {control_genotype})", fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(np.arange(len(metric_order)))
        ax.set_xticklabels(display_metric_labels, rotation=70, ha="right", fontsize=FONT_SIZE_N_TEXT)
        ax.set_yticks(np.arange(len(tested_genotypes)))
        ax.set_yticklabels(tested_genotypes, fontsize=FONT_SIZE_TICKS)

        for row_i, genotype in enumerate(tested_genotypes):
            for col_i, metric in enumerate(metric_order):
                val = matrix.loc[genotype, metric]
                if not np.isfinite(val):
                    continue
                pval = pmat.loc[genotype, metric]
                star = significance_stars(pval) if pd.notna(pval) else ""
                if annotate_values:
                    text_val = f"{val:.2f}"
                    text = f"{text_val}\n{star}" if star and star != "ns" else text_val
                    ax.text(col_i, row_i, text, ha="center", va="center", fontsize=FONT_SIZE_N_TEXT, color="black")
                elif star and star != "ns":
                    ax.text(col_i, row_i, star, ha="center", va="center", fontsize=FONT_SIZE_LABELS, color="black")

        ax.tick_params(axis="both", which="major", length=2, width=0.8)

    if im is None:
        plt.close(fig)
        return

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Cohen's d (genotype - control)", fontsize=FONT_SIZE_LABELS)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)

    fig.suptitle("Training/Test effect-size heatmaps (tested genotypes only)", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    for fmt in PLOT_FORMATS:
        out = output_dir / f"training_test_cohens_d_heatmap.{fmt}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", format=fmt)

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="F1 TNT training-vs-test comparison (permutation only)")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to summary feather dataset")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metrics to analyze")
    parser.add_argument("--all-metrics", action="store_true", help="Analyze all auto-detected continuous metrics")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Bootstrap resamples used for CI columns in permutation statistics CSV",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be >= 1")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("F1 TNT TRAINING-vs-TEST COMPARISON (PERMUTATION ONLY)")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"(Pooled target when ready: {POOLED_DATASET_PATH})")

    df = pd.read_feather(dataset_path)

    genotype_col = detect_column(df, ["Genotype", "genotype"], "genotype")
    pretraining_col = detect_column(df, ["Pretraining", "pretraining"], "pretraining")
    identity_col = detect_column(df, ["ball_identity", "ball_condition"], "ball identity")
    fly_col = detect_column(df, ["fly", "fly_name"], "fly")

    active_genotypes, control_genotype = resolve_active_genotypes(df, genotype_col)
    if not active_genotypes:
        raise ValueError("No genotype values found in dataset")

    if control_genotype != PREFERRED_CONTROL_GENOTYPE:
        print(
            f"Warning: preferred control '{PREFERRED_CONTROL_GENOTYPE}' not found. "
            f"Using '{control_genotype}' as control for this dataset."
        )

    print(f"Active genotypes: {active_genotypes}")
    print(f"Control genotype: {control_genotype}")

    df_filt = filter_pretrained_testable(df, genotype_col, pretraining_col, identity_col, active_genotypes)
    if df_filt.empty:
        raise ValueError(
            "No rows left after filtering for pretrained flies + selected genotypes + training/test identities"
        )

    print(f"Rows after filtering: {len(df_filt)}")
    print("Counts by genotype and ball identity:")
    print(df_filt.groupby([genotype_col, identity_col]).size().unstack(fill_value=0))

    all_available_metrics = auto_detect_metrics(df_filt)
    metric_lookup = {m.lower(): m for m in all_available_metrics}

    if args.metrics:
        requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        metrics = []
        missing_metrics = []
        seen = set()
        for m in requested_metrics:
            resolved = metric_lookup.get(m.lower())
            if resolved is None:
                missing_metrics.append(m)
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            metrics.append(resolved)

        if missing_metrics:
            print("Warning: requested metrics not found after filtering:")
            print(", ".join(missing_metrics))
    elif args.all_metrics:
        metrics = all_available_metrics
    else:
        default_metrics = [m for m in DEFAULT_METRICS_WITH_SUPPLEMENTARY if m in df_filt.columns]
        additional_auto_metrics = [m for m in all_available_metrics if m not in default_metrics]
        metrics = default_metrics + additional_auto_metrics

        if additional_auto_metrics:
            print("Including additional auto-detected metrics not in default list:")
            print(", ".join(additional_auto_metrics[:20]))
            if len(additional_auto_metrics) > 20:
                print(f"... and {len(additional_auto_metrics) - 20} more")

    metrics_for_stats = sorted(set(metrics).union(all_available_metrics))

    print(f"\nAnalyzing {len(metrics)} metrics for plots/scatter")
    print(f"Computing permutation/Cohen's d stats on {len(metrics_for_stats)} metrics (heatmap uses all available)")
    if not metrics:
        raise ValueError("No valid metrics selected")

    stats_rows = []

    for metric in metrics_for_stats:
        # permutation stats by subset (training/test)
        training_stats = compute_permutation_stats(
            df_filt,
            metric,
            genotype_col,
            identity_col,
            subset_identity="training",
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
        )
        test_stats = compute_permutation_stats(
            df_filt,
            metric,
            genotype_col,
            identity_col,
            subset_identity="test",
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
        )

        if not training_stats.empty:
            stats_rows.append(training_stats)
        if not test_stats.empty:
            stats_rows.append(test_stats)

        if metric not in metrics:
            continue

        # plots
        plot_jitter_box(
            df_filt,
            metric,
            genotype_col,
            identity_col,
            subset_identity="training",
            stats_df=training_stats,
            output_dir=out_root / "training_jitterboxplots",
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
            show=args.show,
        )
        plot_jitter_box(
            df_filt,
            metric,
            genotype_col,
            identity_col,
            subset_identity="test",
            stats_df=test_stats,
            output_dir=out_root / "test_jitterboxplots",
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
            show=args.show,
        )
        plot_training_vs_test_scatter(
            df_filt,
            metric,
            fly_col,
            genotype_col,
            identity_col,
            output_dir=out_root / "training_vs_test_scatterplots",
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
            show=args.show,
        )

    # save permutation stats csv
    if stats_rows:
        perm_stats = pd.concat(stats_rows, ignore_index=True)
        perm_csv = out_root / "permutation_statistics_training_and_test.csv"
        perm_stats.to_csv(perm_csv, index=False)
        print(f"Saved: {perm_csv}")

        heatmap_dir = out_root / "effect_size_heatmaps"
        plot_training_test_cohens_d_heatmaps(
            perm_stats=perm_stats,
            all_metrics=all_available_metrics,
            active_genotypes=active_genotypes,
            control_genotype=control_genotype,
            output_dir=heatmap_dir,
            show=args.show,
        )
        print(f"Saved: {heatmap_dir / 'training_test_cohens_d_heatmap.png'}")

    # correlation csv
    corr_df = compute_training_test_correlations(
        df_filt,
        metrics,
        fly_col,
        genotype_col,
        identity_col,
        active_genotypes,
        control_genotype,
    )
    corr_csv = out_root / "training_vs_test_rank_correlation_vs_control.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved: {corr_csv}")

    # backward-compatible filename
    legacy_corr_csv = out_root / "training_vs_test_correlation_r2.csv"
    corr_df.to_csv(legacy_corr_csv, index=False)
    print(f"Saved: {legacy_corr_csv}")

    ranked = corr_df.loc[
        (corr_df["group"] == "control_baseline") & (corr_df["group_genotype"] == control_genotype)
    ].copy()
    ranked_order = np.argsort(ranked["rho2"].to_numpy(dtype=float))[::-1]
    ranked = ranked.iloc[ranked_order]
    ranked_csv = out_root / "training_vs_test_rank_correlation_rho2_ranked.csv"
    ranked.to_csv(ranked_csv, index=False)
    print(f"Saved: {ranked_csv}")

    print(f"\nTop control-baseline rank-correlation metrics ({control_genotype}):")
    print(ranked[["metric", "n", "rho", "rho2", "p_value"]].head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
