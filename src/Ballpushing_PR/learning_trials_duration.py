#!/usr/bin/env python3
"""
Trial Duration Analysis Script

This script analyzes trial durations for flies that complete at least 4 trials.
It performs statistical tests (Friedman test + post-hoc Wilcoxon signed-rank tests)
and generates a boxplot with statistical annotations.

Author: MD
Date: 2025-03-20
"""

import sys
from pathlib import Path


import matplotlib

# matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import datetime


def load_annotated_data(data_path):
    """
    Load the annotated dataset from a feather file.

    Parameters
    ----------
    data_path : Path or str
        Path to the feather file containing annotated data

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    print(f"Loading data from: {data_path}")
    data = feather.read_feather(data_path)
    print(f"Loaded {len(data)} rows with {data['fly'].nunique()} unique flies")
    return data


def filter_flies_with_n_trials(data, n_trials=4):
    """
    Filter dataset to keep only flies that completed at least n trials.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    n_trials : int, optional
        Minimum number of trials required (default: 4)

    Returns
    -------
    pd.DataFrame
        Filtered dataset
    """
    print(f"\nFiltering flies with at least {n_trials} trials...")

    # Keep only trials 1 to n_trials
    data_filtered = data[data["trial"] <= n_trials]

    # Keep only flies that have all n trials
    data_filtered = data_filtered.groupby("fly").filter(lambda x: x["trial"].nunique() >= n_trials)

    n_flies = data_filtered["fly"].nunique()
    print(f"Found {n_flies} flies with at least {n_trials} trials")

    return data_filtered


def compute_trial_durations(data):
    """
    Compute trial duration for each fly and trial.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset with 'fly', 'trial', and 'trial_time' columns

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fly, trial, duration
    """
    print("\nComputing trial durations...")

    def get_duration(group):
        duration = group["trial_time"].max() - group["trial_time"].min()
        return pd.Series({"duration": duration})

    trial_durations = data.groupby(["fly", "trial"]).apply(get_duration).reset_index()

    # Ensure trial is integer
    trial_durations["trial"] = trial_durations["trial"].astype(int)

    print(f"Computed durations for {len(trial_durations)} trial instances")

    return trial_durations


def format_p_value(p_value, decimals=4):
    """
    Format p-value for display.

    Parameters
    ----------
    p_value : float
        P-value to format
    decimals : int
        Number of decimal places (default: 4)

    Returns
    -------
    str : Formatted p-value string
    """
    if p_value < 0.001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.{decimals}f}"


def format_seconds_to_mmss(seconds):
    """
    Format seconds to MM:SS format.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str : Formatted time string in MM:SS format
    """
    minutes = int(abs(seconds) // 60)
    secs = int(abs(seconds) % 60)
    sign = "-" if seconds < 0 else ""
    return f"{sign}{minutes}:{secs:02d}"


def bootstrap_ci_paired_difference(group1_data, group2_data, n_bootstrap=10000, ci=95):
    """
    Calculate bootstrapped confidence interval for mean difference in paired samples.

    Parameters
    ----------
    group1_data : array-like
        Data from first group (baseline)
    group2_data : array-like
        Data from second group
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)

    Returns
    -------
    tuple : (mean_diff, lower_bound, upper_bound, pct_lower, pct_upper)
            where pct values are percentage changes relative to group1 mean
    """
    # Calculate paired differences
    diffs = group2_data - group1_data
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_diffs = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs.append(np.mean(sample_diffs))

    # Calculate CI
    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)
    mean_diff = np.mean(diffs)

    # Calculate percentage change relative to group1 mean
    mean_group1 = np.mean(group1_data)
    if mean_group1 != 0:
        pct_lower = (lower / mean_group1) * 100
        pct_upper = (upper / mean_group1) * 100
    else:
        pct_lower = pct_upper = np.nan

    return mean_diff, lower, upper, pct_lower, pct_upper


def perform_statistical_tests(trial_durations):
    """
    Perform Friedman test and post-hoc Wilcoxon signed-rank tests.

    Parameters
    ----------
    trial_durations : pd.DataFrame
        DataFrame with columns: fly, trial, duration

    Returns
    -------
    tuple
        (friedman_stat, friedman_p, posthoc_results)
    """
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Prepare data in wide format: rows = flies, columns = trials
    pivot = trial_durations.pivot(index="fly", columns="trial", values="duration").dropna()

    print(f"\nData shape for tests: {pivot.shape[0]} flies × {pivot.shape[1]} trials")

    # Friedman test (non-parametric repeated measures test)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])

    print(f"\nFriedman Test:")
    print(f"  χ² = {friedman_stat:.3f}")
    print(f"  p-value = {friedman_p:.3g}")

    # Post-hoc: Wilcoxon signed-rank test for each pair of trials
    print(f"\nPost-hoc Wilcoxon Signed-Rank Tests (FDR corrected):")

    results = []
    for trial1, trial2 in combinations(pivot.columns, 2):
        # Wilcoxon signed-rank test
        wilcoxon_result = stats.wilcoxon(pivot[trial1], pivot[trial2])
        stat = wilcoxon_result.statistic
        p = wilcoxon_result.pvalue

        # Calculate Z-statistic (approximation for large n)
        n = len(pivot)
        z_stat = abs(stat - (n * (n + 1) / 4)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

        # Calculate bootstrapped CI for difference
        mean_diff, ci_lower, ci_upper, pct_ci_lower, pct_ci_upper = bootstrap_ci_paired_difference(
            pivot[trial1].values, pivot[trial2].values
        )

        results.append(
            {
                "trial1": trial1,
                "trial2": trial2,
                "statistic": stat,
                "z_statistic": z_stat,
                "p": p,
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_ci_lower": pct_ci_lower,
                "pct_ci_upper": pct_ci_upper,
                "mean_diff_mmss": format_seconds_to_mmss(mean_diff),
                "ci_lower_mmss": format_seconds_to_mmss(ci_lower),
                "ci_upper_mmss": format_seconds_to_mmss(ci_upper),
            }
        )

    # Multiple testing correction (FDR)
    pvals = [r["p"] for r in results]
    reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_corr"] = pvals_corr[i]
        r["significant"] = reject[i]

        sig_marker = "*" if r["significant"] else ""
        print(f"  Trial {r['trial1']} vs {r['trial2']}: Z = {r['z_statistic']:.3f}, p = {r['p_corr']:.4g} {sig_marker}")

    print("=" * 60 + "\n")

    return friedman_stat, friedman_p, results


def save_statistics_table(trial_durations, friedman_stat, friedman_p, posthoc_results, output_dir):
    """
    Save descriptive and statistical results to markdown and CSV files.

    Parameters
    ----------
    trial_durations : pd.DataFrame
        DataFrame with trial duration data
    friedman_stat : float
        Friedman test statistic
    friedman_p : float
        Friedman test p-value
    posthoc_results : list
        List of dictionaries with post-hoc test results
    output_dir : Path or str
        Directory to save output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute descriptive statistics by trial
    desc_stats = (
        trial_durations.groupby("trial")["duration"]
        .agg(["count", "mean", "std", "median", ("q1", lambda x: x.quantile(0.25)), ("q3", lambda x: x.quantile(0.75))])
        .round(3)
    )

    # Save markdown table
    md_file = output_dir / "trial_duration_statistics.md"
    with open(md_file, "w") as f:
        f.write("# Trial Duration Analysis Results\n\n")
        f.write(f"**Analysis Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Friedman Test\n\n")
        f.write(f"- **Test Statistic (χ²):** {friedman_stat:.3f}\n")
        f.write(f"- **P-value:** {format_p_value(friedman_p)}\n\n")

        f.write("## Descriptive Statistics by Trial\n\n")
        f.write("| Trial | N | Mean (s) | Std (s) | Median (s) | Q1 (s) | Q3 (s) |\n")
        f.write("|-------|---|----------|---------|------------|--------|--------|\n")
        for trial, row in desc_stats.iterrows():
            f.write(
                f"| {int(trial)} | {int(row['count'])} | "
                f"{row['mean']:.3f} | {row['std']:.3f} | {row['median']:.3f} | "
                f"{row['q1']:.3f} | {row['q3']:.3f} |\n"
            )

        f.write("\n## Post-hoc Wilcoxon Signed-Rank Tests (FDR Corrected)\n\n")
        f.write(
            "| Trial 1 | Trial 2 | Z | Mean Diff (s) | 95% CI Lower (s) | 95% CI Upper (s) | 95% CI % Lower | 95% CI % Upper | P-value | P-corr | Sig |\\n"
        )
        f.write(
            "|---------|---------|---|-----------------|------------------|------------------|----------------|----------------|---------|--------|-----|\\n"
        )
        for r in posthoc_results:
            sig_text = "✓" if r["significant"] else "✗"
            f.write(
                f"| {int(r['trial1'])} | {int(r['trial2'])} | {r['z_statistic']:.3f} | {r['mean_diff']:.3f} | "
                f"{r['ci_lower']:.3f} | {r['ci_upper']:.3f} | {r['pct_ci_lower']:.2f}% | {r['pct_ci_upper']:.2f}% | "
                f"{format_p_value(r['p'])} | {format_p_value(r['p_corr'])} | {sig_text} |\n"
            )

    print(f"✅ Statistical results saved to: {md_file}")

    # Save CSV file
    csv_file = output_dir / "trial_duration_statistics.csv"
    with open(csv_file, "w") as f:
        f.write("Trial,N,Mean_s,Std_s,Median_s,Q1_s,Q3_s\n")
        for trial, row in desc_stats.iterrows():
            f.write(
                f"{int(trial)},{int(row['count'])},{row['mean']:.3f},{row['std']:.3f},"
                f"{row['median']:.3f},{row['q1']:.3f},{row['q3']:.3f}\n"
            )

    print(f"✅ Descriptive statistics saved to: {csv_file}")

    # Save post-hoc results CSV
    posthoc_csv = output_dir / "trial_duration_posthoc.csv"
    # Keep only the relevant columns for the CSV
    posthoc_df = pd.DataFrame(posthoc_results)[
        [
            "trial1",
            "trial2",
            "z_statistic",
            "mean_diff",
            "mean_diff_mmss",
            "ci_lower",
            "ci_lower_mmss",
            "ci_upper",
            "ci_upper_mmss",
            "pct_ci_lower",
            "pct_ci_upper",
            "p",
            "p_corr",
            "significant",
        ]
    ]
    posthoc_df.to_csv(posthoc_csv, index=False)
    print(f"✅ Post-hoc test results saved to: {posthoc_csv}")


def create_duration_plot(trial_durations, friedman_stat, friedman_p, posthoc_results, output_dir):
    """
    Create and save trial duration boxplot with statistical annotations.

    Parameters
    ----------
    trial_durations : pd.DataFrame
        DataFrame with trial duration data
    friedman_stat : float
        Friedman test statistic
    friedman_p : float
        Friedman test p-value
    posthoc_results : list
        List of dictionaries with post-hoc test results
    output_dir : Path or str
        Directory to save output plots
    """
    print("Creating plot...")

    plt.figure(figsize=(8, 6))

    # Create boxplot
    sns.boxplot(
        data=trial_durations,
        x="trial",
        y="duration",
        color="black",
        showcaps=False,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        fliersize=0,
    )

    # Add stripplot with jitter (black points)
    sns.stripplot(
        data=trial_durations,
        x="trial",
        y="duration",
        color="black",
        size=4,
        alpha=0.5,
        jitter=0.2,
    )

    # Add connecting lines for individual flies
    for fly in trial_durations["fly"].unique():
        fly_data = trial_durations[trial_durations["fly"] == fly]
        plt.plot(fly_data["trial"].values - 1, fly_data["duration"].values, color="gray", alpha=0.2, linewidth=0.5)

    # Labels and title
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("Trial Duration (s)", fontsize=12)
    plt.title(f"Friedman p={friedman_p:.3g}", fontsize=14)

    # Add statistical annotations for significant pairs
    y_max = trial_durations["duration"].max()
    y_range = trial_durations["duration"].max() - trial_durations["duration"].min()
    y_step = y_range * 0.07
    annot_y = y_max + y_step

    for r in posthoc_results:
        if r["significant"]:
            x1 = int(r["trial1"]) - 1
            x2 = int(r["trial2"]) - 1

            # Draw bracket
            plt.plot(
                [x1, x1, x2, x2], [annot_y, annot_y + y_step / 3, annot_y + y_step / 3, annot_y], color="k", lw=1.5
            )

            # Add significance stars
            if r["p_corr"] < 0.001:
                stars = "***"
            elif r["p_corr"] < 0.01:
                stars = "**"
            elif r["p_corr"] < 0.05:
                stars = "*"
            else:
                stars = ""

            plt.text((x1 + x2) / 2, annot_y + y_step / 3, stars, ha="center", va="bottom", color="k", fontsize=14)

            annot_y += y_step

    plt.tight_layout()

    # Save plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eps_path = output_dir / "Trial_Duration_4trials_withstats_jitter.eps"
    png_path = output_dir / "Trial_Duration_4trials_withstats_jitter.png"
    pdf_path = output_dir / "Trial_Duration_4trials_withstats_jitter.pdf"

    plt.savefig(eps_path, dpi=300, bbox_inches="tight", format="eps")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")

    print(f"Saved plots to:")
    print(f"  - {eps_path}")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")

    plt.show()
    plt.close()


def main():
    """Main analysis pipeline."""

    print("\n" + "=" * 60)
    print("TRIAL DURATION ANALYSIS")
    print("=" * 60 + "\n")

    # Configuration
    data_file = Path(
        "/mnt/upramdya_data/MD/BallPushing_Learning/Datasets/" "250318_Datasets/250320_Annotated_data.feather"
    )
    output_dir = Path("/mnt/upramdya_data/MD/BallPushing_Learning/Plots/Durations")
    min_trials = 4

    # Load data
    annotated_data = load_annotated_data(data_file)

    # Filter flies with sufficient trials
    filtered_data = filter_flies_with_n_trials(annotated_data, n_trials=min_trials)

    # Compute trial durations
    trial_durations = compute_trial_durations(filtered_data)

    # Perform statistical tests
    friedman_stat, friedman_p, posthoc_results = perform_statistical_tests(trial_durations)

    # Create and save plot
    create_duration_plot(trial_durations, friedman_stat, friedman_p, posthoc_results, output_dir)

    # Save statistics tables
    save_statistics_table(trial_durations, friedman_stat, friedman_p, posthoc_results, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
