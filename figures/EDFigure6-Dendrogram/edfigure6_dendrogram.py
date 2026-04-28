#!/usr/bin/env python3
"""
ED Figure 6 — Two-way dendrogram of per-metric effect sizes across high-consistency hits.

Produces:
  • metric_two_way_dendrogram.{png,pdf,svg}  — clustered heatmap / dendrogram
  • statistical_results_detailed.{csv,md}    — per-genotype × per-metric stats table

Derived from src/PCA/plot_detailed_metric_statistics.py — dendrogram + CSV only.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap  # noqa: F401 (kept for parity)
from matplotlib.patches import Patch  # noqa: F401
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    leaves_list,
    linkage,
    set_link_color_palette,
)
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests

from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import set_illustrator_style

# Rebuild font cache if needed
fm._load_fontmanager(try_read_cache=False)
set_illustrator_style()
warnings.filterwarnings("ignore")

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not available - using matplotlib only")

# ``Config`` lives in ``src/Plotting/Config.py`` outside the figures
# tree; load it by path instead of relying on a particular CWD or a
# ``sys.path.append`` so this script runs from any directory (including
# when executed by run_all_figures.py with cwd=script.parent).
import importlib.util as _importlib_util  # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "Plotting" / "Config.py"
_spec = _importlib_util.spec_from_file_location("Config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not locate Config module at {_CONFIG_PATH}")
Config = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(Config)

# ── Configuration ────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = dataset("Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather")
CONSISTENCY_DIR = _REPO_ROOT / "src/Screen_analysis/pca_analysis_results_tailored_20251219_163028/data_files"
METRICS_PATH = _REPO_ROOT / "src/Screen_analysis/metrics_lists/final_metrics_for_pca_alt.txt"
OUTPUT_DIR = figure_output_dir("EDFigure6", __file__, create=False)

MIN_COMBINED_CONSISTENCY = 0.80
CONTROL_MODE = "tailored"  # "tailored" | "emptysplit" | "tnt_pr"
USE_PERMUTATION_PER_PC = True  # True = permutation test; False = Mann-Whitney U
DISCRETE_EFFECTS = False
CLIP_EFFECTS = 1.5

# ── Brain-region look-ups ─────────────────────────────────────────────────────

try:
    nickname_to_brainregion = dict(zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"]))
    color_dict = Config.color_dict
except Exception as e:
    print(f"⚠️  Could not load region mapping from Config: {e}")
    nickname_to_brainregion = {}
    color_dict = {}

# ── Metric display names ──────────────────────────────────────────────────────

METRIC_DISPLAY_NAMES = {
    "pulling_ratio": "Proportion pull vs push",
    "distance_ratio": "Dist. ball moved / corridor length",
    "distance_moved": "Dist. ball moved",
    "pulled": "Signif. (>0.3 mm) pulling events (#)",
    "max_event": "Event max. ball displ. (#)",
    "number_of_pauses": "Long pauses (>5s <5px) (#)",
    "first_major_event": "First major (>1.2mm) event(#)",
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
    "fraction_not_facing_ball": "Fraction not facing (>30°) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "speed_trend": "Slope linear fit to fly speed over time",
}


def get_display_name(metric_name):
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


# ── Statistical helpers ───────────────────────────────────────────────────────


def permutation_test_1d(group1, group2, n_permutations=10000, random_state=None):
    rng = np.random.default_rng(random_state)
    observed = np.abs(group1.mean() - group2.mean())
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(combined))
        perm_combined = combined[perm_idx]
        stat = np.abs(perm_combined[:n1].mean() - perm_combined[n1:].mean())
        if stat >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs[i] = np.mean(s1) - np.mean(s2)
    alpha = 100 - ci
    return float(np.percentile(diffs, alpha / 2)), float(np.percentile(diffs, 100 - alpha / 2))


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bin_cohens_d(cohens_d_value):
    abs_d = abs(cohens_d_value)
    sign = 1 if cohens_d_value >= 0 else -1
    if abs_d < 0.2:
        return 0
    elif abs_d < 0.5:
        return sign * 0.35
    elif abs_d < 1.0:
        return sign * 0.75
    else:
        return sign * 1.5


def format_p_value(p_value):
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.6f}"


def significance_label(p_value):
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ── Axis helpers ──────────────────────────────────────────────────────────────


def colour_y_ticklabels(ax, nickname_to_region, color_dict):
    for tick in ax.get_yticklabels():
        region = nickname_to_region.get(tick.get_text(), None)
        if region in color_dict:
            tick.set_color(color_dict[region])


# ── Data loading ──────────────────────────────────────────────────────────────


def load_consistency_results():
    criteria_file = os.path.join(CONSISTENCY_DIR, "statistical_criteria_comparison.csv")
    if os.path.exists(criteria_file):
        df = pd.read_csv(criteria_file)
        if "Triple_Pass" in df.columns:
            filtered = df[df["Triple_Pass"] == True].copy()
            print("📊 LOADING EXACT REPRODUCED HITS (Triple Test):")
            print(f"   Hits passing Triple test: {len(filtered)}")
            for _, row in filtered.iterrows():
                consistency = row.get("Triple_Consistency_%", 0)
                print(f"   {row['Genotype']:<30}  Consistency={consistency:.1f}%")
            return filtered["Genotype"].tolist()
        print("❌ statistical_criteria_comparison.csv missing Triple_Pass column")
        return []

    # Fallback to combined consistency files
    for fname in ("combined_consistency_ranking.csv", "enhanced_consistency_scores.csv"):
        fpath = os.path.join(CONSISTENCY_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            break
    else:
        print("❌ No consistency file found.")
        return []

    if "Combined_Consistency" not in df.columns:
        if "Overall_Consistency" in df.columns:
            for col in ("Optimized_Only_Consistency", "Optimized_Consistency"):
                if col in df.columns:
                    df["Combined_Consistency"] = 0.5 * df["Overall_Consistency"] + 0.5 * df[col]
                    break
            else:
                print("❌ Cannot derive Combined_Consistency.")
                return []
        else:
            print("❌ Cannot derive Combined_Consistency.")
            return []

    genotype_col = "Genotype" if "Genotype" in df.columns else ("genotype" if "genotype" in df.columns else None)
    if genotype_col is None:
        print("❌ No genotype column.")
        return []

    filtered = df[df["Combined_Consistency"] >= MIN_COMBINED_CONSISTENCY].sort_values(
        "Combined_Consistency", ascending=False
    )

    stats_file = os.path.join(CONSISTENCY_DIR, "static_pca_stats_results_allmethods_tailoredctrls.csv")
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        keep = set(
            stats_df[stats_df["Permutation_FDR_significant"] & stats_df["Mahalanobis_FDR_significant"]][
                "Nickname"
            ].unique()
        )
        filtered = filtered[filtered[genotype_col].isin(keep)]

    print(f"📊 High-consistency hits: {len(filtered)}")
    return filtered[genotype_col].tolist()


def load_metrics_list():
    for path in (
        METRICS_PATH,
        _REPO_ROOT / "src/Screen_analysis/metrics_lists/final_metrics_for_pca.txt",
    ):
        if os.path.exists(path):
            with open(path) as f:
                metrics = [l.strip() for l in f if l.strip()]
            print(f"📋 Loaded {len(metrics)} metrics from {path}")
            return metrics
    print("❌ Metrics file not found.")
    return []


def load_nickname_mapping():
    region_map_path = dataset("Region_map_250908.csv")
    try:
        region_map = pd.read_csv(region_map_path)
        nickname_mapping = dict(zip(region_map["Nickname"], region_map["Simplified Nickname"]))
        simplified_to_region = dict(zip(region_map["Simplified Nickname"], region_map["Simplified region"]))
        print(f"📋 Loaded {len(nickname_mapping)} nickname mappings")
        return nickname_mapping, simplified_to_region
    except Exception as e:
        print(f"⚠️  Could not load region mapping: {e}")
        return {}, {}


def apply_simplified_nicknames(genotype_list, nickname_mapping):
    if not nickname_mapping:
        return genotype_list
    return [nickname_mapping.get(g, g) for g in genotype_list]


# ── Data preparation ──────────────────────────────────────────────────────────


def prepare_data():
    print("📊 Loading and preprocessing data...")
    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)
    exclude = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "MB247-Gal4"]
    dataset = dataset[~dataset["Nickname"].isin(exclude)]
    dataset.rename(
        columns={"major_event": "first_major_event", "major_event_time": "first_major_event_time"},
        inplace=True,
    )
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)
    return dataset


# ── Metric analysis ───────────────────────────────────────────────────────────


def run_metric_analysis(dataset, metrics_list, high_consistency_hits):
    print(f"\n🔬 RUNNING METRIC ANALYSIS — {len(metrics_list)} metrics, {len(high_consistency_hits)} genotypes")
    available_metrics = [m for m in metrics_list if m in dataset.columns]
    binary_metrics = {"has_finished", "has_major", "has_significant"}
    available_metrics = [m for m in available_metrics if m not in binary_metrics]

    na_pct = dataset[available_metrics].isna().sum() / len(dataset) * 100
    valid_metrics = [m for m in available_metrics if na_pct[m] <= 5.0]

    valid_data = dataset[valid_metrics].copy()
    dataset_clean = dataset[~valid_data.isnull().any(axis=1)].copy()
    print(f"   Final data: {dataset_clean.shape[0]} rows, {len(valid_metrics)} metrics")

    scaler = RobustScaler()
    scaler.fit_transform(dataset_clean[valid_metrics].to_numpy())

    metadata_cols = [c for c in dataset_clean.columns if c not in valid_metrics]
    metric_with_meta = pd.concat(
        [dataset_clean[metadata_cols].reset_index(drop=True), dataset_clean[valid_metrics].reset_index(drop=True)],
        axis=1,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_clean[valid_metrics].to_csv(os.path.join(OUTPUT_DIR, "metric_values.csv"), index=False)
    correlation_matrix = dataset_clean[valid_metrics].corr()
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "metric_correlations.csv"))
    metric_with_meta.to_feather(os.path.join(OUTPUT_DIR, "metrics_with_metadata.feather"))

    analysis_genotypes = [g for g in high_consistency_hits if g in metric_with_meta["Nickname"].values]
    print(f"   🎯 Analyzing {len(analysis_genotypes)} genotypes")

    results = []
    for nickname in analysis_genotypes:
        force_control = None
        if CONTROL_MODE == "emptysplit":
            force_control = "Empty-Split"
        elif CONTROL_MODE == "tnt_pr":
            force_control = "TNTxPR"

        subset = Config.get_subset_data(metric_with_meta, col="Nickname", value=nickname, force_control=force_control)
        if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
            continue
        control_names = [n for n in subset["Nickname"].unique() if n != nickname]
        if not control_names:
            continue
        control_name = control_names[0]

        ppc_pvals, metrics_tested, directions, effect_sizes, metric_stats = [], [], {}, {}, {}

        for metric in valid_metrics:
            group_arr = subset[subset["Nickname"] == nickname][metric].values.astype(float)
            control_arr = subset[subset["Nickname"] == control_name][metric].values.astype(float)
            if len(group_arr) == 0 or len(control_arr) == 0:
                continue

            genotype_mean = float(np.mean(group_arr))
            control_mean = float(np.mean(control_arr))
            mean_diff = genotype_mean - control_mean
            ci_lower, ci_upper = bootstrap_ci_difference(group_arr, control_arr, n_bootstrap=10000, random_state=42)
            pct_change = (mean_diff / control_mean * 100) if control_mean != 0 else np.nan

            if USE_PERMUTATION_PER_PC:
                pval = permutation_test_1d(group_arr, control_arr, random_state=42)
            else:
                _, pval = mannwhitneyu(group_arr, control_arr, alternative="two-sided")

            d = cohens_d(group_arr, control_arr)
            ppc_pvals.append(pval)
            metrics_tested.append(metric)
            effect_sizes[metric] = d
            directions[metric] = 1 if d > 0 else -1
            metric_stats[metric] = {
                "genotype_n": len(group_arr),
                "control_n": len(control_arr),
                "genotype_mean": genotype_mean,
                "control_mean": control_mean,
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_change": pct_change,
            }

        if ppc_pvals:
            rejected, pvals_corr, _, _ = multipletests(ppc_pvals, alpha=0.05, method="fdr_bh")
            significant_metrics = [metrics_tested[i] for i, r in enumerate(rejected) if r]
            ppc_any = any(rejected)
        else:
            rejected, pvals_corr, significant_metrics, ppc_any = [], [], [], False

        result_dict = {
            "genotype": nickname,
            "control": control_name,
            "significant_metrics": significant_metrics,
            "num_significant_metrics": len(significant_metrics),
            "significant": ppc_any,
        }
        for i, metric in enumerate(metrics_tested):
            if i < len(ppc_pvals):
                result_dict[f"{metric}_pval"] = ppc_pvals[i]
                result_dict[f"{metric}_pval_corrected"] = pvals_corr[i] if i < len(pvals_corr) else ppc_pvals[i]
                result_dict[f"{metric}_significant"] = bool(rejected[i]) if i < len(rejected) else False
                result_dict[f"{metric}_direction"] = directions.get(metric, 0)
                result_dict[f"{metric}_cohens_d"] = effect_sizes.get(metric, 0.0)
                s = metric_stats.get(metric, {})
                for k in (
                    "genotype_n",
                    "control_n",
                    "genotype_mean",
                    "control_mean",
                    "mean_diff",
                    "ci_lower",
                    "ci_upper",
                    "pct_change",
                ):
                    result_dict[f"{metric}_{k}"] = s.get(k, np.nan)
        results.append(result_dict)

    if not results:
        print("   ⚠️  No statistical results generated")
        return pd.DataFrame(), correlation_matrix

    results_df = pd.DataFrame(results)
    test_method = "permutation" if USE_PERMUTATION_PER_PC else "mannwhitney"
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"metric_stats_results_{test_method}.csv"), index=False)
    print(f"   ✅ Statistical testing complete — {len(results_df)} genotypes")
    return results_df, correlation_matrix


# ── CSV / Markdown statistics table ──────────────────────────────────────────


def save_statistical_results_table(results_df, metrics_list):
    print(f"\n📝 Generating statistical results table...")

    def _fmt_num(v, d=3):
        return "N/A" if v is None or pd.isna(v) else f"{v:.{d}f}"

    def _fmt_pct(v, d=1):
        return "N/A" if v is None or pd.isna(v) else f"{v:+.{d}f}%"

    table_rows = []
    for _, row in results_df.iterrows():
        genotype, control = row["genotype"], row["control"]
        for metric in metrics_list:
            pval_col = f"{metric}_pval"
            if pval_col not in row or pd.isna(row[pval_col]):
                continue
            pval = row[pval_col]
            pval_corrected = row.get(f"{metric}_pval_corrected", pval)
            table_rows.append(
                {
                    "Genotype": genotype,
                    "Control": control,
                    "Metric": get_display_name(metric),
                    "Metric_ID": metric,
                    "Genotype_n": row.get(f"{metric}_genotype_n", np.nan),
                    "Control_n": row.get(f"{metric}_control_n", np.nan),
                    "Genotype_mean": row.get(f"{metric}_genotype_mean", np.nan),
                    "Control_mean": row.get(f"{metric}_control_mean", np.nan),
                    "Mean_diff": row.get(f"{metric}_mean_diff", np.nan),
                    "Bootstrap_CI_lower": row.get(f"{metric}_ci_lower", np.nan),
                    "Bootstrap_CI_upper": row.get(f"{metric}_ci_upper", np.nan),
                    "Percent_change": row.get(f"{metric}_pct_change", np.nan),
                    "P_value": pval,
                    "P_value_formatted": format_p_value(pval),
                    "P_value_corrected": pval_corrected,
                    "P_value_corrected_formatted": format_p_value(pval_corrected),
                    "Cohens_d": row.get(f"{metric}_cohens_d", np.nan),
                    "Significance": significance_label(pval_corrected),
                }
            )

    if not table_rows:
        print("   ⚠️  No statistical results to tabulate")
        return

    stats_table = pd.DataFrame(table_rows).sort_values(["Genotype", "P_value_corrected"])

    csv_path = os.path.join(OUTPUT_DIR, "statistical_results_detailed.csv")
    stats_table.to_csv(csv_path, index=False)
    print(f"   💾 CSV: {csv_path}")

    md_path = os.path.join(OUTPUT_DIR, "statistical_results_detailed.md")
    with open(md_path, "w") as f:
        f.write("# Statistical Results: Detailed Metric Analysis\n\n")
        f.write(f"**Analysis Type:** {'Permutation test' if USE_PERMUTATION_PER_PC else 'Mann-Whitney U test'}\n")
        f.write(f"**FDR Correction:** Benjamini-Hochberg (α = 0.05)\n")
        f.write(f"**Total comparisons:** {len(table_rows)}\n\n")
        for genotype in stats_table["Genotype"].unique():
            gdata = stats_table[stats_table["Genotype"] == genotype]
            ctrlname = gdata["Control"].iloc[0]
            n_sig = (gdata["Significance"] != "ns").sum()
            f.write(f"\n## {genotype} vs {ctrlname}\n\n")
            f.write(f"**Significant metrics:** {n_sig}/{len(gdata)}\n\n")
            f.write(
                "| Metric | n (G/C) | Mean G | Mean C | ΔMean | 95% BS CI | %Δ | P-value | P-value (FDR) | Cohen's d | Sig |\n"
            )
            f.write(
                "|--------|---------|--------|--------|-------|-----------|-----|---------|---------------|-----------|-----|\n"
            )
            for _, mr in gdata.iterrows():
                gn = f"{int(mr['Genotype_n'])}" if not pd.isna(mr["Genotype_n"]) else "N/A"
                cn = f"{int(mr['Control_n'])}" if not pd.isna(mr["Control_n"]) else "N/A"
                ci_text = f"[{_fmt_num(mr.get('Bootstrap_CI_lower'))}, {_fmt_num(mr.get('Bootstrap_CI_upper'))}]"
                d_str = f"{mr['Cohens_d']:.3f}" if not pd.isna(mr["Cohens_d"]) else "N/A"
                f.write(
                    f"| {mr['Metric']} | {gn}/{cn} | {_fmt_num(mr.get('Genotype_mean'))} | "
                    f"{_fmt_num(mr.get('Control_mean'))} | {_fmt_num(mr.get('Mean_diff'))} | {ci_text} | "
                    f"{_fmt_pct(mr.get('Percent_change'))} | {mr['P_value_formatted']} | "
                    f"{mr['P_value_corrected_formatted']} | {d_str} | {mr['Significance']} |\n"
                )
    print(f"   💾 Markdown: {md_path}")

    n_sig_total = (stats_table["Significance"] != "ns").sum()
    print(
        f"\n   📊 {stats_table['Genotype'].nunique()} genotypes, {len(stats_table)} comparisons, "
        f"{n_sig_total} significant ({100*n_sig_total/len(stats_table):.1f}%)"
    )


# ── Matrix builder ────────────────────────────────────────────────────────────


def _collect_metric_columns(results_df, all_metrics):
    cols = [
        c for c in results_df.columns if c.endswith("_significant") and c.replace("_significant", "") in all_metrics
    ]
    return sorted({c.replace("_significant", "") for c in cols})


def _build_signed_weighted_matrix(results_df, all_metrics, only_significant_hits=False, alpha=0.05):
    df = results_df[results_df["significant"]].copy() if only_significant_hits else results_df.copy()
    if df.empty:
        return pd.DataFrame(), []

    metric_names = _collect_metric_columns(results_df, all_metrics)
    if not metric_names:
        return pd.DataFrame(), []

    M = pd.DataFrame(0.0, index=df["genotype"].values, columns=metric_names)
    for _, row in df.iterrows():
        for metric in metric_names:
            cohens_col = f"{metric}_cohens_d"
            d = row.get(cohens_col, 0.0)
            if pd.notna(d) and d != 0:
                if DISCRETE_EFFECTS:
                    M.loc[row["genotype"], metric] = float(bin_cohens_d(d))
                else:
                    if CLIP_EFFECTS is not None:
                        d = np.clip(d, -CLIP_EFFECTS, CLIP_EFFECTS)
                    M.loc[row["genotype"], metric] = float(d)
    return M, metric_names


# ── Two-way dendrogram ────────────────────────────────────────────────────────


def plot_two_way_dendrogram_metrics(
    results_df,
    correlation_matrix,
    nickname_mapping=None,
    simplified_to_region=None,
    only_significant_hits=False,
    alpha=0.05,
    row_metric="euclidean",
    row_linkage="ward",
    col_metric="euclidean",
    col_linkage="ward",
    color_threshold_rows="default",
    color_threshold_cols="default",
    row_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
    col_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
    above_threshold_color_rows="C0",
    above_threshold_color_cols="C0",
    cmap="RdBu_r",
    linewidths=0.4,
    linecolor="gray",
    fig_size=(24, 16),
    row_label_fontsize=9,
    col_label_fontsize=9,
    metric_label_rotation=45,
    annotate=False,
    despine=True,
    row_cluster_level=3,
    cluster_gap_size=8,
):
    cbar_label = "Cohen's d Effect Size (Blue: lower, Red: higher)"
    vmin = -CLIP_EFFECTS if CLIP_EFFECTS is not None else -1.5
    vmax = CLIP_EFFECTS if CLIP_EFFECTS is not None else 1.5

    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df, all_metrics, only_significant_hits=only_significant_hits, alpha=alpha
    )
    if M.empty:
        print("No data to build dendrogram.")
        return None

    print(f"   ✂️  Cohen's d clipped to: [{vmin:.2f}, {vmax:.2f}]")

    # Apply simplified nicknames
    original_to_simplified = {}
    simplified_to_original = {}
    if nickname_mapping:
        original_names = M.index.tolist()
        simplified = apply_simplified_nicknames(original_names, nickname_mapping)
        for orig, simp in zip(original_names, simplified):
            original_to_simplified[orig] = simp
            simplified_to_original[simp] = orig
        M = M.set_axis(simplified, axis=0)
    else:
        for name in M.index:
            simplified_to_original[name] = name

    # Row linkage
    row_Z = linkage(pdist(M.values, metric=row_metric), method=row_linkage) if M.shape[0] > 1 else None

    # Column linkage from correlation-based distance
    corr_subset = correlation_matrix.loc[metric_names, metric_names].fillna(0.0)
    dist_mat = 1 - np.abs(corr_subset.values)
    np.fill_diagonal(dist_mat, 0.0)
    if not np.all(np.isfinite(dist_mat)):
        dist_mat = np.where(np.isfinite(dist_mat), dist_mat, 1.0)
    n = dist_mat.shape[0]
    col_distances = dist_mat[np.triu_indices(n, k=1)]
    if not np.all(np.isfinite(col_distances)):
        col_distances = np.where(np.isfinite(col_distances), col_distances, 1.0)
    col_Z = linkage(col_distances, method=col_linkage) if len(metric_names) > 1 else None

    # Row cluster assignments
    genotypes_by_cluster = {}
    cluster_order = []

    if row_cluster_level is not None and row_Z is not None and M.shape[0] > 1:
        row_cluster_assignments = fcluster(row_Z, row_cluster_level, criterion="maxclust")
        cluster_map = {i: row_cluster_assignments[i] for i in range(len(row_cluster_assignments))}

        set_link_color_palette(list(row_palette))
        dg_row = dendrogram(row_Z, orientation="left", color_threshold=None, no_labels=True, ax=None, no_plot=True)
        row_order_idx = dg_row["leaves"]

        for genotype_idx in row_order_idx:
            cluster_id = cluster_map[genotype_idx]
            if cluster_id not in genotypes_by_cluster:
                genotypes_by_cluster[cluster_id] = []
                cluster_order.append(cluster_id)
            genotypes_by_cluster[cluster_id].append(M.index[genotype_idx])

        print(f"   🔀 {row_cluster_level} clusters: {[len(genotypes_by_cluster[c]) for c in cluster_order]}")
    else:
        genotypes_by_cluster[1] = list(M.index)
        cluster_order = [1]
        row_cluster_level = None

    # Custom display order for 3 clusters: index 2, 0, 1
    num_clusters = len(cluster_order)
    if num_clusters == 3:
        display_order = [cluster_order[2], cluster_order[0], cluster_order[1]]
    else:
        display_order = cluster_order

    cluster_heights = [len(genotypes_by_cluster[cid]) for cid in display_order]
    total_grid_rows = 1 + num_clusters

    plt.style.use("default")
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(
        total_grid_rows,
        4,
        height_ratios=[2.5] + cluster_heights,
        width_ratios=[1.5, 1.2, 8.0, 0.3],
        wspace=0.08,
        hspace=0.15,
    )

    ax_top_dendro = fig.add_subplot(gs[0, 2])
    set_link_color_palette(["#404040"])
    dg_col = dendrogram(col_Z, orientation="top", color_threshold=None, no_labels=True, ax=ax_top_dendro)
    col_order_idx = dg_col["leaves"]
    col_labels_ordered = [metric_names[i] for i in col_order_idx]

    # Recolour top dendrogram lines
    if col_Z is not None and len(metric_names) > 1:
        from matplotlib.collections import LineCollection

        max_dist_top = col_Z[:, 2].max()
        for old_coll in list(ax_top_dendro.collections):
            light_segs, dark_segs = [], []
            for seg in old_coll.get_segments():
                max_y = max(seg[:, 1]) if len(seg) > 0 else 0
                (light_segs if max_y >= max_dist_top * 0.80 else dark_segs).append(seg)
            old_coll.remove()
            if light_segs:
                ax_top_dendro.add_collection(LineCollection(light_segs, colors="lightgray", linewidths=2.0, zorder=8))
            if dark_segs:
                ax_top_dendro.add_collection(LineCollection(dark_segs, colors="#404040", linewidths=2.0, zorder=10))
        for line in ax_top_dendro.get_lines():
            max_y = max(line.get_ydata()) if len(line.get_ydata()) > 0 else 0
            if max_y >= max_dist_top * 0.80:
                line.set_color("lightgray")
                line.set_linewidth(2.0)
                line.set_zorder(8)
            else:
                line.set_color("#404040")
                line.set_linewidth(2.0)
                line.set_zorder(10)
    else:
        dg_col = None
        ax_top_dendro.axis("off")

    # Save canonical metric order
    try:
        canonical_path = (
            Path(__file__).parent.parent.parent / "src" / "PCA" / "metrics_lists" / "canonical_metrics_order.txt"
        )
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with canonical_path.open("w") as fh:
            for m in col_labels_ordered:
                fh.write(f"{m}\n")
        print(f"   💾 Saved canonical metric order to: {canonical_path}")
    except Exception as e:
        print(f"⚠️  Could not save canonical metric order: {e}")

    # Cluster heatmap axes
    ax_hm_list, ax_left_dendro_list, ax_label_list = [], [], []
    for i, cluster_id in enumerate(display_order):
        row_idx = 1 + i
        ax_left_dendro_list.append(fig.add_subplot(gs[row_idx, 0]))
        ax_label_list.append(fig.add_subplot(gs[row_idx, 1]))
        ax_hm_list.append((cluster_id, fig.add_subplot(gs[row_idx, 2])))

    ax_cbar = fig.add_subplot(gs[1 : num_clusters + 1, 3])

    # Build cluster matrices
    cluster_matrices = {}
    for cluster_id in display_order:
        genotypes = genotypes_by_cluster[cluster_id]
        cm = M.loc[genotypes, :]
        cluster_matrices[cluster_id] = cm.iloc[:, col_order_idx] if col_order_idx else cm

    cmap_obj = plt.get_cmap("RdBu_r").copy()
    try:
        cmap_obj.set_bad(alpha=0.0)
    except Exception:
        pass

    for i, (cluster_id, ax_hm) in enumerate(ax_hm_list):
        display_matrix = cluster_matrices[cluster_id]
        ax_label = ax_label_list[i]

        if HAS_SEABORN:
            sns.heatmap(
                display_matrix,
                ax=ax_hm,
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                linewidths=linewidths,
                linecolor=linecolor,
                square=False,
                xticklabels=False,
                yticklabels=False,
                annot=False,
            )
        else:
            ax_hm.imshow(display_matrix.values, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
            ax_hm.set_xticks(np.arange(display_matrix.shape[1] + 1) - 0.5, minor=True)
            ax_hm.set_yticks(np.arange(display_matrix.shape[0] + 1) - 0.5, minor=True)
            ax_hm.grid(which="minor", color=linecolor, linestyle="-", linewidth=linewidths)
            ax_hm.tick_params(which="minor", size=0)

        ax_hm.set_yticks([])
        ax_hm.set_yticklabels([])

        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(-0.5, len(display_matrix.index) - 0.5)
        ax_label.invert_yaxis()
        ax_label.set_xticks([])
        ax_label.yaxis.tick_right()
        ax_label.yaxis.set_label_position("right")
        ax_label.set_yticks(np.arange(len(display_matrix.index)))
        ax_label.set_yticklabels(display_matrix.index, fontsize=row_label_fontsize, ha="right")
        ax_label.tick_params(axis="y", which="both", length=0, pad=2)
        for spine in ax_label.spines.values():
            spine.set_visible(False)

        if i == len(ax_hm_list) - 1:
            ax_hm.set_xticks(np.arange(len(col_labels_ordered)) + 0.5)
            ax_hm.set_xticklabels(
                [get_display_name(m) for m in col_labels_ordered],
                rotation=metric_label_rotation,
                ha="right",
                fontsize=col_label_fontsize,
            )
        else:
            ax_hm.set_xticks([])
            ax_hm.set_xticklabels([])

        # Colour y-tick labels by brain region
        try:
            region_mapping = simplified_to_region if simplified_to_region else nickname_to_brainregion
            colour_y_ticklabels(ax_label, region_mapping, color_dict)
        except Exception as e:
            print(f"   ⚠️  Could not colour y-labels for cluster {cluster_id}: {e}")

        # Significance stars
        for row_idx_cell, genotype in enumerate(display_matrix.index):
            orig = simplified_to_original.get(genotype, genotype)
            grow = results_df[results_df["genotype"] == orig]
            if grow.empty:
                continue
            for j, metric in enumerate(col_labels_ordered):
                pval = grow.iloc[0].get(f"{metric}_pval_corrected", np.nan)
                if pd.isna(pval):
                    continue
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else None
                if stars:
                    try:
                        cell_val = display_matrix.iloc[row_idx_cell, j]
                    except Exception:
                        cell_val = None
                    tc = "white" if (cell_val is not None and abs(cell_val) >= 0.5) else "black"
                    ax_hm.text(
                        j + 0.5,
                        row_idx_cell + 0.5,
                        stars,
                        ha="center",
                        va="center",
                        color=tc,
                        fontsize=11,
                        fontweight="bold",
                    )

    # Left dendrograms
    if row_Z is not None and M.shape[0] > 1:
        set_link_color_palette(["#404040"])
        dg_left = dendrogram(
            row_Z, orientation="left", color_threshold=0, above_threshold_color="#404040", no_labels=True, no_plot=True
        )

        leaf_to_position = {}
        current_pos = 0
        for cluster_id in display_order:
            for genotype in genotypes_by_cluster[cluster_id]:
                leaf_idx = list(M.index).index(genotype)
                leaf_to_position[leaf_idx] = current_pos
                current_pos += 1

        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            cluster_genotypes_list = genotypes_by_cluster[cluster_id]
            cluster_size = len(cluster_genotypes_list)
            cluster_leaf_indices = {list(M.index).index(g) for g in cluster_genotypes_list}

            global_to_local_y = {}
            for j, genotype in enumerate(cluster_genotypes_list):
                global_leaf_idx = list(M.index).index(genotype)
                global_to_local_y[leaf_to_position[global_leaf_idx]] = j

            def get_segment_leaves(icoord_segment):
                leaves = []
                for y_val in icoord_segment:
                    leaf_idx = int((y_val - 5) / 10)
                    if 0 <= leaf_idx < len(dg_left["leaves"]):
                        leaves.append(dg_left["leaves"][leaf_idx])
                return leaves

            max_distance_in_cluster = 0
            for ic, dc in zip(dg_left["icoord"], dg_left["dcoord"]):
                segs = get_segment_leaves(ic)
                if segs and all(leaf in cluster_leaf_indices for leaf in segs):
                    max_distance_in_cluster = max(max_distance_in_cluster, max(dc))

            max_x = 0
            for ic, dc, color in zip(dg_left["icoord"], dg_left["dcoord"], dg_left.get("color_list", [])):
                seg_leaves = get_segment_leaves(ic)
                if not (seg_leaves and all(leaf in cluster_leaf_indices for leaf in seg_leaves)):
                    continue
                new_ic, valid = [], True
                for y_val in ic:
                    leaf_idx = int((y_val - 5) / 10)
                    if 0 <= leaf_idx < len(dg_left["leaves"]):
                        actual_leaf = dg_left["leaves"][leaf_idx]
                        global_pos = leaf_to_position[actual_leaf]
                        if global_pos in global_to_local_y:
                            new_ic.append(global_to_local_y[global_pos])
                        else:
                            valid = False
                            break
                    else:
                        frac = max(0, min((y_val - 5) / 10, len(dg_left["leaves"]) - 1))
                        lo, hi = int(np.floor(frac)), min(int(np.floor(frac)) + 1, len(dg_left["leaves"]) - 1)
                        ll, hl = dg_left["leaves"][lo], dg_left["leaves"][hi]
                        if ll in cluster_leaf_indices and hl in cluster_leaf_indices:
                            lly = global_to_local_y.get(leaf_to_position[ll], 0)
                            hly = global_to_local_y.get(leaf_to_position[hl], cluster_size - 1)
                            new_ic.append(lly * (1 - (frac - lo)) + hly * (frac - lo))
                        else:
                            valid = False
                            break

                if valid and len(new_ic) == 4:
                    is_root = max_distance_in_cluster > 0 and max(dc) >= max_distance_in_cluster * 0.90
                    ax_left.plot(
                        dc,
                        new_ic,
                        color="lightgray" if is_root else "#404040",
                        linewidth=2.0,
                        zorder=8 if is_root else 10,
                    )
                    max_x = max(max_x, max(dc))

            ax_left.set_ylim(-0.5, cluster_size - 0.5)
            ax_left.invert_xaxis()
            ax_left.invert_yaxis()
            ax_left.set_xticks([])
            ax_left.set_yticks([])
            for spine in ax_left.spines.values():
                spine.set_visible(False)
            ax_left.set_clip_on(True)

            # "Group N" label above each cluster's dendrogram
            ax_left.text(
                0.5,
                1.02,
                f"Group {i + 1}",
                transform=ax_left.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )

        # Inter-cluster stem lines
        from matplotlib.lines import Line2D

        cluster_roots = []
        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            cluster_leaf_indices = {list(M.index).index(g) for g in genotypes_by_cluster[cluster_id]}
            max_d = 0
            for dc, ic in zip(dg_left["dcoord"], dg_left["icoord"]):
                segs = []
                for y_val in ic:
                    li = int((y_val - 5) / 10)
                    if 0 <= li < len(dg_left["leaves"]):
                        segs.append(dg_left["leaves"][li])
                if segs and all(l in cluster_leaf_indices for l in segs):
                    max_d = max(max_d, max(dc))
            cluster_roots.append(
                {"ax": ax_left, "y_pos": (len(genotypes_by_cluster[cluster_id]) - 1) / 2.0, "max_dist": max_d}
            )

        stem_x = max(cr["max_dist"] for cr in cluster_roots) * 1.15
        ax_ref = cluster_roots[0]["ax"]
        pt = ax_ref.transData.transform([stem_x, cluster_roots[0]["y_pos"]])
        stem_x_fig = fig.transFigure.inverted().transform(pt)[0]

        root_endpoints = []
        for cr in cluster_roots:
            pt2 = cr["ax"].transData.transform([cr["max_dist"], cr["y_pos"]])
            root_endpoints.append(fig.transFigure.inverted().transform(pt2))

        for i, re in enumerate(root_endpoints):
            fig.add_artist(
                Line2D(
                    [re[0], stem_x_fig],
                    [re[1], re[1]],
                    color="lightgray",
                    linewidth=2.0,
                    transform=fig.transFigure,
                    zorder=5,
                )
            )
        if len(root_endpoints) > 1:
            fig.add_artist(
                Line2D(
                    [stem_x_fig, stem_x_fig],
                    [root_endpoints[0][1], root_endpoints[-1][1]],
                    color="lightgray",
                    linewidth=2.0,
                    transform=fig.transFigure,
                    zorder=5,
                )
            )
    else:
        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            ax_left.text(
                0.5,
                0.5,
                f"C{cluster_id}",
                transform=ax_left.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            ax_left.axis("off")

    # Colorbar
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    pos = ax_cbar.get_position()
    nh = pos.height * 0.35
    ax_cbar.set_position((pos.x0, pos.y0 + (pos.height - nh) / 2, pos.width, nh))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=8)

    try:
        thresh = CLIP_EFFECTS if CLIP_EFFECTS is not None else 1.5
        base_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ticks = sorted({-thresh} | set(base_ticks) | {thresh})
        tick_labels = [
            f"≤ {t:.2f}" if np.isclose(t, -thresh) else f"≥ {t:.2f}" if np.isclose(t, thresh) else f"{t:.1f}"
            for t in ticks
        ]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    except Exception:
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.ax.tick_params(labelsize=7)

    # "Metric" label above the column dendrogram
    ax_top_dendro.text(
        0.5,
        1.04,
        "Metric",
        transform=ax_top_dendro.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )

    # Clean up top dendrogram
    ax_top_dendro.set_xticks([])
    ax_top_dendro.set_yticks([])
    for spine in ax_top_dendro.spines.values():
        spine.set_visible(False)
    ax_top_dendro.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if despine:
        for _, ax_hm in ax_hm_list:
            if HAS_SEABORN:
                sns.despine(ax=ax_hm, top=True, right=True, left=False, bottom=False)
            else:
                ax_hm.spines["top"].set_visible(False)
                ax_hm.spines["right"].set_visible(False)

    # "Driver line" vertical label alongside the inter-cluster stem
    all_left_pos = [ax.get_position() for ax in ax_left_dendro_list]
    y_top = all_left_pos[0].y1
    y_bot = all_left_pos[-1].y0
    x_label = all_left_pos[0].x0 * 0.45  # to the left of the dendrogram column
    fig.text(
        x_label,
        (y_top + y_bot) / 2,
        "Driver line",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
        fontweight="bold",
        transform=fig.transFigure,
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300, "bbox_inches": "tight"} if ext == "png" else {"bbox_inches": "tight"}
        if ext == "svg":
            kwargs["format"] = "svg"
        fig.savefig(os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.{ext}"), **kwargs)
    plt.close(fig)

    # Save matrices / orders
    M.to_csv(os.path.join(OUTPUT_DIR, "metric_two_way_matrix.csv"))
    ordered_genotypes = [g for cid in display_order for g in genotypes_by_cluster[cid]]
    pd.Series(ordered_genotypes, name="genotype").to_csv(
        os.path.join(OUTPUT_DIR, "metric_two_way_row_order.csv"), index=False
    )
    pd.Series(col_labels_ordered, name="metric").to_csv(
        os.path.join(OUTPUT_DIR, "metric_two_way_col_order.csv"), index=False
    )
    if row_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, "metric_two_way_row_linkage.npy"), row_Z)
    if col_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, "metric_two_way_col_linkage.npy"), col_Z)

    print(f"   💾 Saved dendrogram figures and supporting data to {OUTPUT_DIR}")
    return {
        "matrix": M,
        "row_Z": row_Z,
        "col_Z": col_Z,
        "row_order": ordered_genotypes,
        "col_order": col_labels_ordered,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("ED FIGURE 6 — TWO-WAY DENDROGRAM")
    print("=" * 50)

    high_consistency_hits = load_consistency_results()
    if not high_consistency_hits:
        print("❌ No high-consistency hits found.")
        return

    metrics_list = load_metrics_list()
    if not metrics_list:
        print("❌ No metrics loaded.")
        return

    dataset = prepare_data()
    results_df, correlation_matrix = run_metric_analysis(dataset, metrics_list, high_consistency_hits)

    if results_df.empty:
        print("❌ No results generated.")
        return

    save_statistical_results_table(results_df, metrics_list)

    nickname_mapping, simplified_to_region = load_nickname_mapping()

    print(f"\n🎨 Creating two-way dendrogram...")
    plot_two_way_dendrogram_metrics(
        results_df,
        correlation_matrix,
        nickname_mapping=nickname_mapping,
        simplified_to_region=simplified_to_region,
        only_significant_hits=False,
        alpha=0.05,
        row_metric="euclidean",
        row_linkage="ward",
        col_metric="euclidean",
        col_linkage="ward",
        fig_size=(24, 16),
        annotate=False,
        row_cluster_level=3,
        cluster_gap_size=8,
    )

    print(f"\n✅ ED Figure 6 complete — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
