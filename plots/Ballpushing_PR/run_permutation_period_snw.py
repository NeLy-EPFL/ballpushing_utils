#!/usr/bin/env python3
"""
Script to generate permutation-test jitterboxplots for starved_noWater flies,
comparing AM vs PM period.

This script generates permutation-test visualizations with:
- starved_noWater FeedingState only
- Light ON condition only (filtered from dataset)
- Period as grouping variable: AM vs PM (PM15 folded into PM)
- Short experiments excluded (< 2 h recordings)
- Single pairwise comparison: AM vs PM
- Statistical significance annotation (*, **, ***)

Usage:
    python run_permutation_period_snw.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: Skip metrics that already have plots.
    --test:         Process only first 3 metrics with a small sample (debugging).
"""
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from statsmodels.stats.multitest import multipletests
from ballpushing_utils import read_feather


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Experiments excluded because recording duration is ~1 h instead of 2 h.
SHORT_EXPERIMENTS = [
    "240718_Afternoon_FeedingState_Videos_Tracked",
    "240718_Afternoon_FeedingState_next_Videos_Tracked",
]

PERIOD_ORDER = ["AM", "PM"]
PERIOD_LABELS = {"AM": "AM", "PM": "PM"}
PERIOD_COLORS = {
    "AM": "#E6AB02",  # golden yellow
    "PM": "#7570B3",  # purple
}

TIME_METRICS = {
    "max_event_time",
    "final_event_time",
    "first_significant_event_time",
    "first_major_event_time",
    "chamber_exit_time",
    "chamber_time",
    "time_chamber_beginning",
}


# ---------------------------------------------------------------------------
# Core plotting + statistics function
# ---------------------------------------------------------------------------


def generate_period_permutation_plots(
    data,
    metrics,
    y="Period",
    control_condition="AM",
    palette=None,
    output_dir="permutation_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
    n_permutations=10000,
):
    """
    Generate jitterboxplots for each metric with a permutation test comparing
    AM vs PM period within starved_noWater flies.

    Returns
    -------
    pd.DataFrame
        Statistics table with permutation test results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    all_conditions = [c for c in PERIOD_ORDER if c in data[y].unique()]
    print(f"Period conditions found: {all_conditions}")

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"\nPlot {metric_idx + 1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y]).copy()
        sorted_conditions = [c for c in PERIOD_ORDER if c in plot_data[y].unique()]

        if len(sorted_conditions) < 2:
            print(f"  Only {len(sorted_conditions)} condition(s) present – skipping.")
            continue

        # --- Permutation test: AM vs PM ---
        vals_am = plot_data[plot_data[y] == "AM"][metric].dropna()
        vals_pm = plot_data[plot_data[y] == "PM"][metric].dropna()

        if len(vals_am) == 0 or len(vals_pm) == 0:
            print(f"  Missing data for one condition – skipping.")
            continue

        try:
            obs_stat = float(np.median(vals_pm) - np.median(vals_am))
            combined = np.concatenate([vals_am.values, vals_pm.values])
            n_am = len(vals_am)

            perm_diffs = np.empty(n_permutations)
            for i in range(n_permutations):
                np.random.shuffle(combined)
                perm_diffs[i] = np.median(combined[n_am:]) - np.median(combined[:n_am])

            pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))
            direction = "increased" if obs_stat > 0 else ("decreased" if obs_stat < 0 else "none")
            test_name = f"permutation({n_permutations})"
        except Exception as e:
            print(f"  Error: {e}")
            pval = 1.0
            obs_stat = np.nan
            direction = "none"
            test_name = f"permutation({n_permutations}) (failed)"

        significant = pval < alpha
        if pval < 0.001:
            sig_level = "***"
        elif pval < 0.01:
            sig_level = "**"
        elif pval < 0.05:
            sig_level = "*"
        else:
            sig_level = "ns"

        result = {
            "ConditionA": "AM",
            "ConditionB": "PM",
            "Metric": metric,
            "pval_raw": pval,
            "pval_corrected": pval,  # only one comparison per metric
            "significant": significant,
            "sig_level": sig_level,
            "direction": direction if significant else "none",
            "effect_size": obs_stat,
            "median_a": vals_am.median(),
            "median_b": vals_pm.median(),
            "mean_a": vals_am.mean(),
            "mean_b": vals_pm.mean(),
            "n_a": len(vals_am),
            "n_b": len(vals_pm),
            "test_type": test_name,
        }
        all_stats.append(result)

        # --- Figure ---
        fig_width = 120 / 25.4
        fig_height = 110 / 25.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        font_size_ticks = 9
        font_size_labels = 13
        font_size_annotations = 14

        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_conditions, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        for idx, cond in enumerate(sorted_conditions):
            cond_data = plot_data[plot_data[y] == cond][metric].dropna()
            if len(cond_data) == 0:
                continue
            color = PERIOD_COLORS.get(cond, "gray")
            linestyle = "solid" if cond == control_condition else "dashed"
            ax.boxplot(
                [cond_data],
                positions=[idx],
                widths=0.55,
                patch_artist=False,
                showfliers=False,
                vert=True,
                boxprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                whiskerprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                capprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                medianprops=dict(color="black", linewidth=2.5, linestyle=linestyle),
            )
            x_jitter = np.random.normal(idx, 0.07, size=len(cond_data))
            if cond == control_condition:
                ax.scatter(
                    x_jitter,
                    cond_data,
                    s=30,
                    facecolors=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.55,
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_jitter, cond_data, s=30, facecolors="none", edgecolors=color, linewidths=0.8, alpha=0.65, zorder=3
                )

        counts = [len(plot_data[plot_data[y] == c]) for c in sorted_conditions]
        display_labels = [f"{PERIOD_LABELS.get(c, c)}\n(n={cnt})" for c, cnt in zip(sorted_conditions, counts)]
        ax.set_xticks(range(len(sorted_conditions)))
        ax.set_xticklabels(display_labels, fontsize=font_size_ticks)

        # Y limits + optional significance bracket
        y_max = plot_data[metric].quantile(0.99)
        y_min = plot_data[metric].min()
        data_range = max(y_max - y_min, 1e-6)
        ax.set_ylim(bottom=y_min - 0.05 * data_range, top=y_max + 0.18 * data_range)

        if significant:
            ann_y = y_max + 0.06 * data_range
            ax.annotate("", xy=(1, ann_y), xytext=(0, ann_y), arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
            ax.text(
                0.5,
                ann_y,
                sig_level,
                fontsize=font_size_annotations,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
            )

        display_metric = metric.replace("_", " ")
        if metric in TIME_METRICS:
            display_metric = f"{display_metric} (min)"

        ax.set_ylabel(display_metric, fontsize=font_size_labels)
        ax.set_xlabel("Period", fontsize=font_size_labels)
        plt.yticks(fontsize=font_size_ticks)
        ax.grid(False)
        ax.legend_ = None

        plt.tight_layout()

        output_file = output_dir / f"{metric}_period_snw_permutation.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {output_file}  [{sig_level}  p={pval:.4f}]")
        print(f"  Time: {time.time() - metric_start_time:.2f}s")

    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        stats_file = output_dir / "period_snw_permutation_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved: {stats_file}")

        total = len(stats_df)
        sig = int(stats_df["significant"].sum())
        print(f"{total} metrics | Significant: {sig} ({100 * sig / total:.1f}%)")

    return stats_df


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load dataset, filter for starved_noWater + Light ON, normalise Period."""
    dataset_path = (
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
        "260220_10_summary_control_folders_Data/summary/pooled_summary.feather"
    )
    print(f"Loading dataset from: {dataset_path}")
    dataset = read_feather(dataset_path)
    print(f"Loaded. Shape: {dataset.shape}")

    # Filter Light ON
    dataset = dataset[dataset["Light"] == "on"].copy()
    print(f"After Light=on filter: {dataset.shape}")

    # Exclude short experiments
    if "experiment" in dataset.columns:
        before = len(dataset)
        dataset = dataset[~dataset["experiment"].isin(SHORT_EXPERIMENTS)].copy()
        dropped = before - len(dataset)
        if dropped:
            print(f"Excluded {dropped} rows from short experiments.")

    # Filter starved_noWater
    dataset["FeedingState"] = dataset["FeedingState"].str.strip().str.lower()
    dataset = dataset[dataset["FeedingState"] == "starved_nowater"].copy()
    print(f"After starved_noWater filter: {dataset.shape}")

    if len(dataset) == 0:
        raise ValueError("No starved_noWater rows found after filtering.")

    # Normalise Period: anything with AM → AM, anything with PM → PM
    if "Period" not in dataset.columns:
        raise ValueError("'Period' column not found.")
    dataset["Period"] = dataset["Period"].str.strip()
    dataset["Period"] = dataset["Period"].apply(
        lambda p: (
            "AM"
            if isinstance(p, str) and "AM" in p.upper()
            else ("PM" if isinstance(p, str) and "PM" in p.upper() else None)
        )
    )
    dataset = dataset[dataset["Period"].isin(["AM", "PM"])].copy()
    print(f"Period values:\n{dataset['Period'].value_counts()}")

    # Drop unneeded metadata
    metadata_columns = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Genotype",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
        "Orientation",
        "Crossing",
        "BallType",
        "Used_to ",
        "Magnet",
        "Peak",
        "Light",
        "FeedingState",
    ]
    dataset = dataset.drop(columns=[c for c in metadata_columns if c in dataset.columns])

    # Boolean → int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns
    drop_cols = [c for c in ["insight_effect", "insight_effect_log", "exit_time", "index"] if c in dataset.columns]
    if drop_cols:
        dataset.drop(columns=drop_cols, inplace=True)

    print(f"Dataset ready. Final shape: {dataset.shape}")

    if test_mode and len(dataset) > test_sample_size:
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"TEST MODE: sampled {test_sample_size} rows.")

    return dataset


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def categorize_metric(col, dataset):
    if col not in dataset.columns:
        return None
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        return "non_numeric"
    non_nan = dataset[col].dropna()
    if len(non_nan) < 3:
        return "insufficient_data"
    unique_vals = set(non_nan.unique())
    if len(unique_vals) == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
        return "binary"
    if len(unique_vals) < 3:
        return "categorical"
    return "continuous"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(overwrite=True, test_mode=False):
    print("Starting Period (AM vs PM) permutation analysis for starved_noWater flies...")

    base_output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/Summary_metrics/Permutation")
    condition_dir = base_output_dir / "permutation_period_snw"
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {condition_dir}")

    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Convert time metrics to minutes
    for m in TIME_METRICS:
        if m in dataset.columns:
            dataset[m] = dataset[m] / 60.0
            print(f"  Converted to min: {m}")

    print(f"\nDataset shape: {dataset.shape}")
    print(f"Period distribution:\n{dataset['Period'].value_counts()}")

    # Metric discovery
    excluded_patterns = [
        "binned_",
        "r2",
        "slope",
        "_bin_",
        "logistic_",
        "learning_",
        "interaction_rate_bin",
        "binned_auc",
        "binned_slope",
    ]
    base_metrics = [
        "nb_events",
        "max_event",
        "max_event_time",
        "max_distance",
        "final_event",
        "final_event_time",
        "nb_significant_events",
        "significant_ratio",
        "first_significant_event",
        "first_significant_event_time",
        "first_major_event",
        "first_major_event_time",
        "major_event_first",
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_speed",
        "auc",
        "overall_interaction_rate",
        "chamber_time",
        "pushed",
    ]
    additional_patterns = [
        "speed",
        "speed",
        "pause",
        "freeze",
        "persistence",
        "trend",
        "interaction_rate",
        "finished",
        "chamber",
        "facing",
        "flailing",
        "head",
        "leg_visibility",
    ]

    available_metrics = []
    for m in base_metrics:
        if m in dataset.columns and not any(p in m for p in excluded_patterns):
            available_metrics.append(m)
    for pat in additional_patterns:
        for col in dataset.columns:
            if pat in col.lower() and col not in available_metrics and col != "Period":
                if not any(p in col.lower() for p in excluded_patterns):
                    available_metrics.append(col)

    continuous_metrics, binary_metrics = [], []
    for m in available_metrics:
        cat = categorize_metric(m, dataset)
        if cat in ("continuous", "insufficient_data"):
            continuous_metrics.append(m)
        elif cat == "binary":
            binary_metrics.append(m)

    print(f"\nContinuous metrics: {len(continuous_metrics)}")
    print(f"Binary metrics: {len(binary_metrics)}")

    if test_mode:
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2]
        print(f"TEST MODE: {len(continuous_metrics)} continuous, {len(binary_metrics)} binary.")

    all_metrics = continuous_metrics + binary_metrics
    if not all_metrics:
        print("No metrics to process. Exiting.")
        return

    required_cols = ["Period"] + all_metrics
    required_cols = [c for c in required_cols if c in dataset.columns]
    analysis_data = dataset[required_cols].copy()

    print(f"\nRunning permutation analysis on {len(all_metrics)} metrics...")
    print(f"Analysis data shape: {analysis_data.shape}")

    t0 = time.time()
    stats_df = generate_period_permutation_plots(
        data=analysis_data,
        metrics=all_metrics,
        y="Period",
        control_condition="AM",
        output_dir=str(condition_dir),
        fdr_method="fdr_bh",
        alpha=0.05,
        n_permutations=10000,
    )
    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.1f}s ({elapsed / 60:.1f} min).")
    print(f"Results saved to: {condition_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Permutation-test jitterboxplots: starved_noWater AM vs PM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 metrics, small sample.")
    args = parser.parse_args()
    main(overwrite=not args.no_overwrite, test_mode=args.test)
