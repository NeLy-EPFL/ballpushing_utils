#!/usr/bin/env python3
"""
Power analysis for TNT MB247 distance_moved metric with multiple comparisons.

This script performs statistical power analysis to determine how many additional
samples are needed to detect a significant effect (if one exists) for the
distance_moved metric in the TNT MB247 experiment.

The script handles multiple comparisons:
1. Within each genotype: Compare pretrained (y) vs non-pretrained (n)
2. Within each pretraining condition: Compare genotypes (TNTxMB247 vs controls)

Multiple testing correction (Bonferroni) is applied to adjust the significance level.

Usage:
    python power_analysis_tnt_mb247.py [--alpha 0.05] [--desired-power 0.8] [--correction bonferroni]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def calculate_effect_size_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for the two groups

    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    mean_diff = np.mean(group1) - np.mean(group2)

    if pooled_std == 0:
        return 0

    return mean_diff / pooled_std


def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d value

    Returns
    -------
    str
        Interpretation of effect size
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_effect_size(group1, group2, n_bootstrap=10000):
    """
    Calculate bootstrap confidence interval for effect size.

    Parameters
    ----------
    group1, group2 : array-like
        Data for the two groups
    n_bootstrap : int
        Number of bootstrap iterations

    Returns
    -------
    tuple
        (mean_effect_size, lower_ci, upper_ci)
    """
    effect_sizes = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)

        d = calculate_effect_size_cohens_d(sample1, sample2)
        effect_sizes.append(d)

    effect_sizes = np.array(effect_sizes)
    mean_d = np.mean(effect_sizes)
    ci_lower = np.percentile(effect_sizes, 2.5)
    ci_upper = np.percentile(effect_sizes, 97.5)

    return mean_d, ci_lower, ci_upper


def power_analysis_mann_whitney(group1, group2, alpha=0.05, desired_power=0.8, n_simulations=1000):
    """
    Perform power analysis for Mann-Whitney U test using simulation.

    This simulates increasing sample sizes and calculates the power to detect
    the observed effect size.

    Parameters
    ----------
    group1, group2 : array-like
        Current data for the two groups
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8)
    n_simulations : int
        Number of simulations per sample size (default: 1000)

    Returns
    -------
    dict
        Dictionary with power analysis results
    """
    # Calculate current statistics
    current_n1 = len(group1)
    current_n2 = len(group2)

    # Test range of sample sizes (from current to 5x current)
    max_multiplier = 5
    sample_sizes = list(range(current_n1, current_n1 * max_multiplier, max(1, current_n1 // 10)))

    if not sample_sizes:
        sample_sizes = [current_n1]

    powers = []

    print(f"\n🔬 Running power analysis simulations...")
    print(f"   Current sample sizes: Group1={current_n1}, Group2={current_n2}")
    print(f"   Testing sample sizes from {min(sample_sizes)} to {max(sample_sizes)}")

    for n in sample_sizes:
        n_significant = 0

        # Simulate n_simulations experiments with sample size n
        for _ in range(n_simulations):
            # Sample with replacement to create datasets of size n
            sim_group1 = np.random.choice(group1, size=n, replace=True)
            sim_group2 = np.random.choice(group2, size=n, replace=True)

            # Perform Mann-Whitney U test
            try:
                _, p_value = mannwhitneyu(sim_group1, sim_group2, alternative="two-sided")
                if p_value < alpha:
                    n_significant += 1
            except:
                pass

        power = n_significant / n_simulations
        powers.append(power)

        # Print progress for key milestones
        if power >= desired_power and len([p for p in powers if p >= desired_power]) == 1:
            print(f"   ✓ Desired power ({desired_power}) first achieved at n={n}")

    # Find sample size needed for desired power
    target_n = None
    for n, power in zip(sample_sizes, powers):
        if power >= desired_power:
            target_n = n
            break

    return {
        "sample_sizes": sample_sizes,
        "powers": powers,
        "current_n1": current_n1,
        "current_n2": current_n2,
        "target_n": target_n,
        "desired_power": desired_power,
        "alpha": alpha,
    }


def plot_power_analysis(power_results, effect_size, output_dir):
    """
    Create comprehensive power analysis visualization.

    Parameters
    ----------
    power_results : dict
        Results from power_analysis_mann_whitney
    effect_size : float
        Observed Cohen's d effect size
    output_dir : Path
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Power curve
    ax1 = axes[0]
    sample_sizes = power_results["sample_sizes"]
    powers = power_results["powers"]
    current_n = power_results["current_n1"]
    target_n = power_results["target_n"]
    desired_power = power_results["desired_power"]

    ax1.plot(sample_sizes, powers, "b-", linewidth=2, label="Statistical Power")
    ax1.axhline(y=desired_power, color="r", linestyle="--", linewidth=2, label=f"Desired Power ({desired_power})")
    ax1.axvline(x=current_n, color="green", linestyle="--", linewidth=2, label=f"Current Sample Size (n={current_n})")

    if target_n:
        ax1.axvline(x=target_n, color="orange", linestyle="--", linewidth=2, label=f"Target Sample Size (n={target_n})")
        ax1.plot(target_n, desired_power, "ro", markersize=10)

    ax1.set_xlabel("Sample Size per Group", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Statistical Power", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Power Analysis for Mann-Whitney U Test\nEffect Size (Cohen's d) = {effect_size:.3f} ({interpret_cohens_d(effect_size)})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim([0, 1.05])

    # Add current power annotation
    current_power_idx = 0  # First element is current sample size
    current_power = powers[current_power_idx]
    ax1.annotate(
        f"Current Power: {current_power:.3f}",
        xy=(current_n, current_power),
        xytext=(current_n + (max(sample_sizes) - min(sample_sizes)) * 0.1, current_power - 0.1),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=10,
        fontweight="bold",
        color="green",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
    )

    # Plot 2: Sample size needed vs desired power
    ax2 = axes[1]
    power_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    required_ns = []

    for power_level in power_levels:
        n_needed = None
        for n, p in zip(sample_sizes, powers):
            if p >= power_level:
                n_needed = n
                break
        required_ns.append(n_needed if n_needed else max(sample_sizes))

    ax2.bar(range(len(power_levels)), required_ns, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axhline(y=current_n, color="green", linestyle="--", linewidth=2, label=f"Current n={current_n}")
    ax2.set_xticks(range(len(power_levels)))
    ax2.set_xticklabels([f"{p:.0%}" for p in power_levels])
    ax2.set_xlabel("Desired Statistical Power", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Required Sample Size per Group", fontsize=12, fontweight="bold")
    ax2.set_title("Sample Size Requirements\nfor Different Power Levels", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (power, n) in enumerate(zip(power_levels, required_ns)):
        color = "red" if n > current_n else "green"
        ax2.text(
            i,
            n + max(required_ns) * 0.02,
            f"{n}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "power_analysis_distance_moved.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Power analysis plot saved to: {output_path}")

    plt.show()


def generate_report(data, metric, power_results, effect_size, ci_lower, ci_upper, output_dir):
    """
    Generate a comprehensive text report of the power analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Original data
    metric : str
        Name of the metric analyzed
    power_results : dict
        Results from power analysis
    effect_size : float
        Cohen's d effect size
    ci_lower, ci_upper : float
        95% confidence interval for effect size
    output_dir : Path
        Directory to save report
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("POWER ANALYSIS REPORT: TNT MB247 - Distance Moved")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Current sample summary
    report_lines.append("## Current Sample Summary")
    report_lines.append("-" * 80)

    # Get groups
    genotypes = data["Genotype"].unique()
    pretraining_vals = data["Pretraining"].unique()

    report_lines.append(f"Metric analyzed: {metric}")
    report_lines.append(f"Genotypes in data: {', '.join(map(str, genotypes))}")
    report_lines.append(f"Pretraining conditions: {', '.join(map(str, pretraining_vals))}")
    report_lines.append("")

    for genotype in genotypes:
        for pretrain in pretraining_vals:
            subset = data[(data["Genotype"] == genotype) & (data["Pretraining"] == pretrain)]
            n = len(subset)
            mean = subset[metric].mean()
            std = subset[metric].std()
            median = subset[metric].median()

            report_lines.append(f"  {genotype} + Pretraining={pretrain}:")
            report_lines.append(f"    n = {n}")
            report_lines.append(f"    Mean ± SD = {mean:.2f} ± {std:.2f}")
            report_lines.append(f"    Median = {median:.2f}")
            report_lines.append("")

    # Effect size analysis
    report_lines.append("## Effect Size Analysis")
    report_lines.append("-" * 80)
    report_lines.append(f"Observed Cohen's d: {effect_size:.3f}")
    report_lines.append(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    report_lines.append(f"Interpretation: {interpret_cohens_d(effect_size).upper()}")
    report_lines.append("")

    # Effect size interpretation guide
    report_lines.append("Effect Size Interpretation Guide:")
    report_lines.append("  - Negligible: |d| < 0.2")
    report_lines.append("  - Small:      0.2 ≤ |d| < 0.5")
    report_lines.append("  - Medium:     0.5 ≤ |d| < 0.8")
    report_lines.append("  - Large:      |d| ≥ 0.8")
    report_lines.append("")

    # Current statistical test results
    report_lines.append("## Current Statistical Test Results")
    report_lines.append("-" * 80)

    # Perform current test
    current_n1 = power_results["current_n1"]
    current_n2 = power_results["current_n2"]
    current_power = power_results["powers"][0] if power_results["powers"] else 0

    report_lines.append(f"Sample sizes: n1={current_n1}, n2={current_n2}")
    report_lines.append(f"Current statistical power: {current_power:.3f} ({current_power*100:.1f}%)")
    report_lines.append(f"Significance level (alpha): {power_results['alpha']}")
    report_lines.append("")

    # Power analysis results
    report_lines.append("## Power Analysis Results")
    report_lines.append("-" * 80)
    report_lines.append(
        f"Desired statistical power: {power_results['desired_power']} ({power_results['desired_power']*100:.0f}%)"
    )

    target_n = power_results["target_n"]
    if target_n:
        additional_needed = target_n - current_n1
        report_lines.append(f"✓ Target sample size per group: {target_n}")
        report_lines.append(f"✓ Additional samples needed per group: {additional_needed}")
        report_lines.append(f"✓ Total additional samples needed (both groups): {additional_needed * 2}")
    else:
        report_lines.append(f"✗ Desired power not achievable with tested sample sizes")
        report_lines.append(
            f"✗ Consider: (1) Effect may be too small, or (2) Increase sample size beyond {max(power_results['sample_sizes'])}"
        )

    report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("-" * 80)

    if abs(effect_size) < 0.2:
        report_lines.append("⚠️  NEGLIGIBLE EFFECT SIZE detected")
        report_lines.append("   The observed effect is very small. Consider:")
        report_lines.append("   1. The biological effect may genuinely be negligible")
        report_lines.append("   2. Reassess experimental design or measurement methods")
        report_lines.append("   3. Very large sample sizes would be needed to detect this effect")
    elif abs(effect_size) < 0.5:
        report_lines.append("📊 SMALL EFFECT SIZE detected")
        report_lines.append("   Moderate sample sizes needed. Consider:")
        report_lines.append(
            f"   1. Collect approximately {additional_needed if target_n else 'many more'} samples per group"
        )
        report_lines.append("   2. Verify measurement precision and experimental conditions")
    elif abs(effect_size) < 0.8:
        report_lines.append("📈 MEDIUM EFFECT SIZE detected")
        report_lines.append("   Reasonable sample sizes should be sufficient:")
        report_lines.append(f"   1. Target sample size: {target_n if target_n else 'See plot'} per group")
        report_lines.append("   2. Continue data collection as planned")
    else:
        report_lines.append("🎯 LARGE EFFECT SIZE detected")
        report_lines.append("   Strong effect - relatively small samples needed:")
        report_lines.append(f"   1. Target sample size: {target_n if target_n else current_n1} per group")
        report_lines.append("   2. Current sample may already be sufficient")

    report_lines.append("")

    # Sample size table
    report_lines.append("## Sample Size Requirements for Different Power Levels")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Power Level':<15} {'Required n':<15} {'Additional Needed':<20} {'Status':<15}")
    report_lines.append("-" * 80)

    power_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for power_level in power_levels:
        n_needed = None
        for n, p in zip(power_results["sample_sizes"], power_results["powers"]):
            if p >= power_level:
                n_needed = n
                break

        if n_needed:
            additional = n_needed - current_n1
            status = "✓ Achieved" if additional <= 0 else "  Needed"
            report_lines.append(f"{power_level:<15.0%} {n_needed:<15} {additional:<20} {status:<15}")
        else:
            report_lines.append(f"{power_level:<15.0%} {'N/A':<15} {'N/A':<20} {'Not achievable':<15}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "power_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {report_path}")


def create_multi_comparison_plots(all_results, corrected_alpha, desired_power, output_dir):
    """
    Create comprehensive visualization for multiple comparisons.

    Parameters
    ----------
    all_results : list
        List of dictionaries with power analysis results for each comparison
    corrected_alpha : float
        Corrected significance level
    desired_power : float
        Target power level
    output_dir : Path
        Directory to save plots
    """
    n_comparisons = len(all_results)

    # Figure 1: Effect sizes and current power for all comparisons
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1a: Effect sizes with confidence intervals
    comparison_labels = [r["comparison"] for r in all_results]
    effect_sizes = [r["effect_size"] for r in all_results]
    ci_lowers = [r["effect_size_ci_lower"] for r in all_results]
    ci_uppers = [r["effect_size_ci_upper"] for r in all_results]

    y_pos = np.arange(len(comparison_labels))
    colors = ["steelblue" if r["type"] == "within_genotype" else "coral" for r in all_results]

    ax1.barh(y_pos, effect_sizes, color=colors, alpha=0.7, edgecolor="black")
    ax1.errorbar(
        effect_sizes,
        y_pos,
        xerr=[
            [effect_sizes[i] - ci_lowers[i] for i in range(len(effect_sizes))],
            [ci_uppers[i] - effect_sizes[i] for i in range(len(effect_sizes))],
        ],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=2,
    )

    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax1.axvline(x=0.2, color="green", linestyle=":", linewidth=1, alpha=0.5, label="Small effect")
    ax1.axvline(x=-0.2, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axvline(x=0.5, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="Medium effect")
    ax1.axvline(x=-0.5, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axvline(x=0.8, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Large effect")
    ax1.axvline(x=-0.8, color="red", linestyle=":", linewidth=1, alpha=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comparison_labels, fontsize=9)
    ax1.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight="bold")
    ax1.set_title("Effect Sizes for All Comparisons\nwith 95% Confidence Intervals", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.legend(loc="lower right", fontsize=8)

    # Add value labels
    for i, (d, label) in enumerate(zip(effect_sizes, comparison_labels)):
        ax1.text(d, i, f"  {d:.3f}", va="center", fontsize=8, fontweight="bold")

    # Plot 1b: Current power vs desired power
    current_powers = [r["current_power"] for r in all_results]

    ax2.barh(y_pos, current_powers, color=colors, alpha=0.7, edgecolor="black", label="Current Power")
    ax2.axvline(x=desired_power, color="red", linestyle="--", linewidth=2, label=f"Desired Power ({desired_power})")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(comparison_labels, fontsize=9)
    ax2.set_xlabel("Statistical Power", fontsize=12, fontweight="bold")
    ax2.set_title(f"Current Statistical Power\n(α = {corrected_alpha:.6f})", fontsize=14, fontweight="bold")
    ax2.set_xlim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(loc="lower right", fontsize=10)

    # Add value labels and additional samples needed
    for i, (power, result) in enumerate(zip(current_powers, all_results)):
        ax2.text(power, i, f"  {power:.2f}", va="center", fontsize=8, fontweight="bold")

        target_n = result["target_n"]
        current_n = result["n1"]
        if target_n and target_n > current_n:
            additional = target_n - current_n
            ax2.text(0.02, i, f"+{additional}", va="center", fontsize=7, style="italic", color="red", fontweight="bold")

    # Add legend for comparison types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, edgecolor="black", label="Within Genotype (n vs y)"),
        Patch(facecolor="coral", alpha=0.7, edgecolor="black", label="Between Genotypes"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_path1 = output_dir / "multi_comparison_overview.png"
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    print(f"\n📊 Overview plot saved to: {plot_path1}")

    # Figure 2: Sample size requirements
    fig2, ax = plt.subplots(figsize=(14, max(8, n_comparisons * 0.6)))

    current_ns = [r["n1"] for r in all_results]
    target_ns = [r["target_n"] if r["target_n"] else r["n1"] * 5 for r in all_results]
    additional_ns = [target - current for target, current in zip(target_ns, current_ns)]

    y_pos = np.arange(len(comparison_labels))

    # Stacked bar: current + additional needed
    ax.barh(y_pos, current_ns, color="lightgreen", alpha=0.7, edgecolor="black", label="Current Sample Size")
    ax.barh(
        y_pos,
        additional_ns,
        left=current_ns,
        color="lightcoral",
        alpha=0.7,
        edgecolor="black",
        label="Additional Needed",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_labels, fontsize=9)
    ax.set_xlabel("Sample Size per Group", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Sample Size Requirements to Achieve {desired_power:.0%} Power\n(α = {corrected_alpha:.6f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=10)

    # Add value labels
    for i, (current, additional, target) in enumerate(zip(current_ns, additional_ns, target_ns)):
        ax.text(current / 2, i, f"{current}", va="center", ha="center", fontsize=9, fontweight="bold")
        if additional > 0:
            ax.text(
                current + additional / 2,
                i,
                f"+{additional}",
                va="center",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="darkred",
            )
        ax.text(target + max(target_ns) * 0.02, i, f"  Total: {target}", va="center", fontsize=8, style="italic")

    plt.tight_layout()
    plot_path2 = output_dir / "multi_comparison_sample_sizes.png"
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    print(f"📊 Sample size plot saved to: {plot_path2}")

    # Figure 3: Individual power curves for each comparison (grid layout)
    n_rows = (n_comparisons + 1) // 2
    n_cols = min(2, n_comparisons)

    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_comparisons == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (result, ax) in enumerate(zip(all_results, axes)):
        power_results = result["power_results"]
        sample_sizes = power_results["sample_sizes"]
        powers = power_results["powers"]
        current_n = result["n1"]
        target_n = result["target_n"]

        ax.plot(sample_sizes, powers, "b-", linewidth=2, label="Statistical Power")
        ax.axhline(y=desired_power, color="r", linestyle="--", linewidth=2, label=f"Desired Power ({desired_power})")
        ax.axvline(x=current_n, color="green", linestyle="--", linewidth=2, label=f"Current n={current_n}")

        if target_n:
            ax.axvline(x=target_n, color="orange", linestyle="--", linewidth=2, label=f"Target n={target_n}")
            ax.plot(target_n, desired_power, "ro", markersize=10)

        ax.set_xlabel("Sample Size per Group", fontsize=10)
        ax.set_ylabel("Statistical Power", fontsize=10)
        ax.set_title(f'{result["comparison"]}\nd={result["effect_size"]:.3f}', fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Power Curves for All Comparisons", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plot_path3 = output_dir / "multi_comparison_power_curves.png"
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    print(f"📊 Power curves plot saved to: {plot_path3}")

    plt.show()


def generate_multi_comparison_report(
    all_results, metric, alpha, corrected_alpha, desired_power, correction, n_comparisons, output_dir
):
    """
    Generate comprehensive text report for multiple comparisons.

    Parameters
    ----------
    all_results : list
        List of dictionaries with results for each comparison
    metric : str
        Name of the metric analyzed
    alpha : float
        Original significance level
    corrected_alpha : float
        Corrected significance level
    desired_power : float
        Target power level
    correction : str
        Correction method used
    n_comparisons : int
        Number of comparisons
    output_dir : Path
        Directory to save report
    """
    report_lines = []

    report_lines.append("=" * 100)
    report_lines.append("POWER ANALYSIS REPORT: TNT MB247 - Distance Moved (Multiple Comparisons)")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Summary
    report_lines.append("## ANALYSIS SUMMARY")
    report_lines.append("-" * 100)
    report_lines.append(f"Metric analyzed: {metric}")
    report_lines.append(f"Number of comparisons: {n_comparisons}")
    report_lines.append(f"Multiple testing correction: {correction.upper()}")
    report_lines.append(f"Original significance level (α): {alpha}")
    report_lines.append(f"Corrected significance level (α): {corrected_alpha:.6f}")
    report_lines.append(f"Desired statistical power: {desired_power} ({desired_power*100:.0f}%)")
    report_lines.append("")

    # Comparison types
    within_genotype = [r for r in all_results if r["type"] == "within_genotype"]
    between_genotype = [r for r in all_results if r["type"] == "between_genotype"]

    report_lines.append("## COMPARISON TYPES")
    report_lines.append("-" * 100)
    report_lines.append(f"Within-genotype comparisons (n vs y): {len(within_genotype)}")
    for r in within_genotype:
        report_lines.append(f"  - {r['comparison']}")
    report_lines.append("")
    report_lines.append(f"Between-genotype comparisons (exp vs ctrl): {len(between_genotype)}")
    for r in between_genotype:
        report_lines.append(f"  - {r['comparison']}")
    report_lines.append("")

    # Detailed results for each comparison
    report_lines.append("## DETAILED RESULTS FOR EACH COMPARISON")
    report_lines.append("=" * 100)

    for i, result in enumerate(all_results, 1):
        report_lines.append(f"\n### COMPARISON {i}: {result['comparison']}")
        report_lines.append("-" * 100)

        # Sample information
        report_lines.append(f"\nSample Information:")
        report_lines.append(f"  Group 1: {result['group1_name']}")
        report_lines.append(f"    n = {result['n1']}")
        report_lines.append(f"    Mean ± SD = {result['mean1']:.2f} ± {result['std1']:.2f}")
        report_lines.append(f"  Group 2: {result['group2_name']}")
        report_lines.append(f"    n = {result['n2']}")
        report_lines.append(f"    Mean ± SD = {result['mean2']:.2f} ± {result['std2']:.2f}")
        report_lines.append("")

        # Effect size
        report_lines.append(f"Effect Size:")
        report_lines.append(
            f"  Cohen's d = {result['effect_size']:.3f} ({interpret_cohens_d(result['effect_size']).upper()})"
        )
        report_lines.append(f"  95% CI: [{result['effect_size_ci_lower']:.3f}, {result['effect_size_ci_upper']:.3f}]")
        report_lines.append("")

        # Statistical test
        report_lines.append(f"Statistical Test (Mann-Whitney U):")
        report_lines.append(f"  p-value = {result['p_value']:.6f}")
        report_lines.append(
            f"  Significant at α={corrected_alpha:.6f}: {'YES ✓' if result['p_value'] < corrected_alpha else 'NO ✗'}"
        )
        report_lines.append("")

        # Power analysis
        report_lines.append(f"Power Analysis:")
        report_lines.append(f"  Current power = {result['current_power']:.3f} ({result['current_power']*100:.1f}%)")

        target_n = result["target_n"]
        current_n = result["n1"]
        if target_n:
            additional = target_n - current_n
            report_lines.append(f"  Target sample size = {target_n} per group")
            if additional > 0:
                report_lines.append(f"  Additional needed = {additional} per group ({additional * 2} total)")
                report_lines.append(f"  Status: NEEDS MORE SAMPLES ⚠️")
            else:
                report_lines.append(f"  Status: ADEQUATE ✓")
        else:
            report_lines.append(f"  Target sample size = Not achievable with tested range")
            report_lines.append(f"  Status: EFFECT TOO SMALL OR MORE SAMPLES NEEDED ❌")

        report_lines.append("")

    # Overall summary
    report_lines.append("\n" + "=" * 100)
    report_lines.append("## OVERALL SUMMARY")
    report_lines.append("=" * 100)

    # Count significant comparisons
    significant = [r for r in all_results if r["p_value"] < corrected_alpha]
    non_significant = [r for r in all_results if r["p_value"] >= corrected_alpha]

    report_lines.append(f"\nSignificant comparisons (p < {corrected_alpha:.6f}): {len(significant)}/{n_comparisons}")
    for r in significant:
        report_lines.append(f"  ✓ {r['comparison']} (p={r['p_value']:.6f}, d={r['effect_size']:.3f})")

    report_lines.append(f"\nNon-significant comparisons: {len(non_significant)}/{n_comparisons}")
    for r in non_significant:
        report_lines.append(f"  ✗ {r['comparison']} (p={r['p_value']:.6f}, d={r['effect_size']:.3f})")

    # Sample size recommendations
    report_lines.append("\n" + "-" * 100)
    report_lines.append("SAMPLE SIZE RECOMMENDATIONS")
    report_lines.append("-" * 100)

    total_additional = 0
    for r in all_results:
        if r["target_n"]:
            additional = max(0, r["target_n"] - r["n1"])
            total_additional += additional * 2  # Both groups
            if additional > 0:
                report_lines.append(f"  {r['comparison']}: +{additional} per group (+{additional*2} total)")

    if total_additional > 0:
        report_lines.append(f"\nTotal additional samples recommended: {total_additional}")
        report_lines.append(f"This will bring all comparisons to {desired_power:.0%} power at α={corrected_alpha:.6f}")
    else:
        report_lines.append("\n✓ All comparisons have adequate power with current sample sizes")

    # Key takeaways
    report_lines.append("\n" + "=" * 100)
    report_lines.append("## KEY TAKEAWAYS")
    report_lines.append("=" * 100)

    # Categorize by effect size
    large_effects = [r for r in all_results if abs(r["effect_size"]) >= 0.8]
    medium_effects = [r for r in all_results if 0.5 <= abs(r["effect_size"]) < 0.8]
    small_effects = [r for r in all_results if 0.2 <= abs(r["effect_size"]) < 0.5]
    negligible_effects = [r for r in all_results if abs(r["effect_size"]) < 0.2]

    if large_effects:
        report_lines.append(f"\n🎯 LARGE effects detected ({len(large_effects)} comparisons):")
        for r in large_effects:
            report_lines.append(f"  • {r['comparison']} (d={r['effect_size']:.3f})")
        report_lines.append("  → These should be detectable with relatively small samples")

    if medium_effects:
        report_lines.append(f"\n📈 MEDIUM effects detected ({len(medium_effects)} comparisons):")
        for r in medium_effects:
            report_lines.append(f"  • {r['comparison']} (d={r['effect_size']:.3f})")
        report_lines.append("  → Moderate sample sizes recommended")

    if small_effects:
        report_lines.append(f"\n📊 SMALL effects detected ({len(small_effects)} comparisons):")
        for r in small_effects:
            report_lines.append(f"  • {r['comparison']} (d={r['effect_size']:.3f})")
        report_lines.append("  → Larger sample sizes needed to achieve adequate power")

    if negligible_effects:
        report_lines.append(f"\n⚠️  NEGLIGIBLE effects detected ({len(negligible_effects)} comparisons):")
        for r in negligible_effects:
            report_lines.append(f"  • {r['comparison']} (d={r['effect_size']:.3f})")
        report_lines.append("  → Very large sample sizes would be needed (may not be worthwhile)")

    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)

    # Print and save
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "power_analysis_multi_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n📄 Comprehensive report saved to: {report_path}")

    # Also save as CSV for easy import
    csv_data = []
    for r in all_results:
        csv_data.append(
            {
                "Comparison": r["comparison"],
                "Type": r["type"],
                "Group1": r["group1_name"],
                "Group2": r["group2_name"],
                "n1": r["n1"],
                "n2": r["n2"],
                "Mean1": r["mean1"],
                "Mean2": r["mean2"],
                "SD1": r["std1"],
                "SD2": r["std2"],
                "Cohens_d": r["effect_size"],
                "Effect_CI_Lower": r["effect_size_ci_lower"],
                "Effect_CI_Upper": r["effect_size_ci_upper"],
                "p_value": r["p_value"],
                "Significant": "Yes" if r["p_value"] < corrected_alpha else "No",
                "Current_Power": r["current_power"],
                "Target_n": r["target_n"] if r["target_n"] else "N/A",
                "Additional_Needed": r["target_n"] - r["n1"] if r["target_n"] and r["target_n"] > r["n1"] else 0,
            }
        )

    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / "power_analysis_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"📊 Results table saved to: {csv_path}")


def main(alpha=0.05, desired_power=0.8, correction="bonferroni"):
    """
    Main function to run power analysis with multiple comparisons.

    Parameters
    ----------
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8 = 80%)
    correction : str
        Multiple testing correction method ('bonferroni' or 'none')
    """
    print("\n" + "=" * 80)
    print("TNT MB247 POWER ANALYSIS: distance_moved (Multiple Comparisons)")
    print("=" * 80)

    # Load data
    dataset_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251104_09_F1_coordinates_F1_TNT_MB247_Data/summary/pooled_summary.feather"
    )

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    print(f"\n📁 Loading dataset from: {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Detect column names
    metric = "distance_moved"

    # Find genotype column
    genotype_col = None
    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
            break

    # Find pretraining column
    pretraining_col = None
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    # Find ball condition column
    ball_condition_col = None
    for col in ["ball_condition", "ball_identity"]:
        if col in df.columns:
            ball_condition_col = col
            break

    if not all([genotype_col, pretraining_col]):
        print(f"❌ Error: Could not find required columns")
        print(f"   Genotype column: {genotype_col}")
        print(f"   Pretraining column: {pretraining_col}")
        return

    if metric not in df.columns:
        print(f"❌ Error: Metric '{metric}' not found in dataset")
        print(f"   Available columns: {', '.join(df.columns)}")
        return

    print(f"\n📊 Detected columns:")
    print(f"   Genotype: {genotype_col}")
    print(f"   Pretraining: {pretraining_col}")
    print(f"   Ball condition: {ball_condition_col}")
    print(f"   Metric: {metric}")

    # Filter for test ball only
    if ball_condition_col:
        test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
        df_test = pd.DataFrame()
        for val in test_ball_values:
            subset = df[df[ball_condition_col] == val]
            if not subset.empty:
                df_test = pd.concat([df_test, subset])

        if not df_test.empty:
            df = df_test
            print(f"✓ Filtered for test ball only: {len(df)} rows")

    # Clean data
    df_clean = df[[genotype_col, pretraining_col, metric]].dropna()
    df_clean = df_clean[~np.isinf(df_clean[metric])]

    print(f"✓ Clean data: {len(df_clean)} rows")

    # Get unique genotypes and pretraining values
    genotypes = sorted(df_clean[genotype_col].unique())
    pretraining_vals = sorted(df_clean[pretraining_col].unique())

    print(f"\n🔬 Groups in data:")
    print(f"   Genotypes: {genotypes}")
    print(f"   Pretraining: {pretraining_vals}")

    # Identify control genotypes (those with "Empty" in the name)
    control_genotypes = [g for g in genotypes if "Empty" in g]
    experimental_genotypes = [g for g in genotypes if "Empty" not in g]

    print(f"\n📋 Group classification:")
    print(f"   Control genotypes: {control_genotypes}")
    print(f"   Experimental genotypes: {experimental_genotypes}")

    # Define all pairwise comparisons
    comparisons = []

    # Type 1: Within each genotype, compare pretraining conditions (n vs y)
    if len(pretraining_vals) >= 2:
        for genotype in genotypes:
            genotype_data = df_clean[df_clean[genotype_col] == genotype]
            groups_available = genotype_data[pretraining_col].unique()

            if len(groups_available) >= 2:
                for i, pt1 in enumerate(pretraining_vals[:-1]):
                    for pt2 in pretraining_vals[i + 1 :]:
                        group1_data = genotype_data[genotype_data[pretraining_col] == pt1][metric].values
                        group2_data = genotype_data[genotype_data[pretraining_col] == pt2][metric].values

                        if len(group1_data) >= 2 and len(group2_data) >= 2:
                            comparisons.append(
                                {
                                    "type": "within_genotype",
                                    "genotype": genotype,
                                    "group1_name": f"{genotype} + {pt1}",
                                    "group2_name": f"{genotype} + {pt2}",
                                    "group1_data": group1_data,
                                    "group2_data": group2_data,
                                    "description": f"{genotype}: {pt1} vs {pt2}",
                                }
                            )

    # Type 2: Within each pretraining condition, compare genotypes
    # Compare experimental vs each control
    if len(genotypes) >= 2:
        for pretrain in pretraining_vals:
            pretrain_data = df_clean[df_clean[pretraining_col] == pretrain]

            # Compare each experimental genotype vs pooled controls
            for exp_geno in experimental_genotypes:
                exp_data = pretrain_data[pretrain_data[genotype_col] == exp_geno][metric].values

                # Get data from all controls
                ctrl_data_list = []
                for ctrl_geno in control_genotypes:
                    ctrl_subset = pretrain_data[pretrain_data[genotype_col] == ctrl_geno][metric].values
                    if len(ctrl_subset) > 0:
                        ctrl_data_list.append(ctrl_subset)

                if ctrl_data_list and len(exp_data) >= 2:
                    ctrl_data_combined = np.concatenate(ctrl_data_list)

                    if len(ctrl_data_combined) >= 2:
                        ctrl_names = " + ".join(control_genotypes)
                        comparisons.append(
                            {
                                "type": "between_genotype",
                                "pretraining": pretrain,
                                "group1_name": f"{exp_geno} + {pretrain}",
                                "group2_name": f"Controls + {pretrain}",
                                "group1_data": exp_data,
                                "group2_data": ctrl_data_combined,
                                "description": f"{pretrain}: {exp_geno} vs Controls",
                            }
                        )

    n_comparisons = len(comparisons)

    if n_comparisons == 0:
        print("\n❌ No valid comparisons found with sufficient data")
        return

    print(f"\n🔍 Found {n_comparisons} comparisons to analyze:")
    for i, comp in enumerate(comparisons, 1):
        print(f"   {i}. {comp['description']}")
        print(f"      - {comp['group1_name']}: n={len(comp['group1_data'])}")
        print(f"      - {comp['group2_name']}: n={len(comp['group2_data'])}")

    # Apply multiple testing correction
    if correction == "bonferroni":
        corrected_alpha = alpha / n_comparisons
        print(f"\n⚙️  Multiple testing correction: Bonferroni")
        print(f"   Original α = {alpha}")
        print(f"   Number of comparisons = {n_comparisons}")
        print(f"   Corrected α = {corrected_alpha:.6f}")
    else:
        corrected_alpha = alpha
        print(f"\n⚙️  No multiple testing correction applied")
        print(f"   α = {alpha}")

    # Analyze each comparison
    all_results = []

    for i, comp in enumerate(comparisons, 1):
        print(f"\n{'='*80}")
        print(f"COMPARISON {i}/{n_comparisons}: {comp['description']}")
        print(f"{'='*80}")

        group1 = comp["group1_data"]
        group2 = comp["group2_data"]

        # Calculate effect size
        effect_size = calculate_effect_size_cohens_d(group1, group2)
        mean_d, ci_lower, ci_upper = bootstrap_effect_size(group1, group2, n_bootstrap=5000)

        print(f"\n📏 Effect Size:")
        print(f"   Cohen's d = {effect_size:.3f} ({interpret_cohens_d(effect_size)})")
        print(f"   95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Current test
        stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        print(f"\n📊 Current Statistical Test (Mann-Whitney U):")
        print(f"   U-statistic = {stat:.2f}")
        print(f"   p-value = {p_value:.6f}")
        print(
            f"   Significant at corrected α={corrected_alpha:.6f}: {'YES ✓' if p_value < corrected_alpha else 'NO ✗'}"
        )

        # Power analysis
        print(f"\n⚡ Running power analysis...")
        power_results = power_analysis_mann_whitney(
            group1,
            group2,
            alpha=corrected_alpha,  # Use corrected alpha
            desired_power=desired_power,
            n_simulations=1000,
        )

        all_results.append(
            {
                "comparison": comp["description"],
                "type": comp["type"],
                "group1_name": comp["group1_name"],
                "group2_name": comp["group2_name"],
                "n1": len(group1),
                "n2": len(group2),
                "mean1": np.mean(group1),
                "mean2": np.mean(group2),
                "std1": np.std(group1),
                "std2": np.std(group2),
                "effect_size": effect_size,
                "effect_size_ci_lower": ci_lower,
                "effect_size_ci_upper": ci_upper,
                "p_value": p_value,
                "current_power": power_results["powers"][0] if power_results["powers"] else 0,
                "target_n": power_results["target_n"],
                "power_results": power_results,
            }
        )

    # Create output directory
    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/power_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive visualizations
    create_multi_comparison_plots(all_results, corrected_alpha, desired_power, output_dir)

    # Generate comprehensive report
    generate_multi_comparison_report(
        all_results, metric, alpha, corrected_alpha, desired_power, correction, n_comparisons, output_dir
    )

    print("\n" + "=" * 80)
    print("✅ Power analysis complete for all comparisons!")
    print("=" * 80)

    # For power analysis, compare pretrained vs non-pretrained within a genotype
    # Or compare genotypes within a pretraining condition
    # Let's do both and pick the comparison with more data

    # Option 1: Compare pretraining within TNTxMB247
    if "TNTxMB247" in genotypes and len(pretraining_vals) >= 2:
        genotype_subset = df_clean[df_clean[genotype_col] == "TNTxMB247"]
        group1 = genotype_subset[genotype_subset[pretraining_col] == pretraining_vals[0]][metric].values
        group2 = genotype_subset[genotype_subset[pretraining_col] == pretraining_vals[1]][metric].values
        comparison_name = f"TNTxMB247: {pretraining_vals[0]} vs {pretraining_vals[1]}"
    elif len(genotypes) >= 2 and len(pretraining_vals) >= 1:
        # Option 2: Compare genotypes within a pretraining condition
        pretrain_subset = df_clean[df_clean[pretraining_col] == pretraining_vals[0]]
        group1 = pretrain_subset[pretrain_subset[genotype_col] == genotypes[0]][metric].values
        group2 = pretrain_subset[pretrain_subset[genotype_col] == genotypes[1]][metric].values
        comparison_name = f"{pretraining_vals[0]}: {genotypes[0]} vs {genotypes[1]}"
    else:
        print("❌ Error: Not enough groups for comparison")
        return

    if len(group1) < 2 or len(group2) < 2:
        print(f"❌ Error: Insufficient data for comparison")
        print(f"   Group 1: n={len(group1)}")
        print(f"   Group 2: n={len(group2)}")
        return

    print(f"\n🎯 Performing power analysis for: {comparison_name}")
    print(f"   Group 1: n={len(group1)}, mean={np.mean(group1):.2f}, std={np.std(group1):.2f}")
    print(f"   Group 2: n={len(group2)}, mean={np.mean(group2):.2f}, std={np.std(group2):.2f}")

    # Calculate effect size
    effect_size = calculate_effect_size_cohens_d(group1, group2)
    print(f"\n📏 Observed Cohen's d: {effect_size:.3f} ({interpret_cohens_d(effect_size)})")

    # Bootstrap confidence interval for effect size
    print(f"\n🔄 Calculating bootstrap confidence interval for effect size...")
    mean_d, ci_lower, ci_upper = bootstrap_effect_size(group1, group2, n_bootstrap=10000)
    print(f"✓ 95% CI for Cohen's d: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # Perform current statistical test
    stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    print(f"\n📊 Current Mann-Whitney U test:")
    print(f"   U-statistic: {stat:.2f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Significant at α={alpha}: {'YES ✓' if p_value < alpha else 'NO ✗'}")

    # Perform power analysis
    print(f"\n⚡ Running power analysis...")
    power_results = power_analysis_mann_whitney(
        group1, group2, alpha=alpha, desired_power=desired_power, n_simulations=1000
    )

    # Create output directory
    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/power_analysis")

    # Prepare data for report
    report_data = pd.DataFrame(
        {
            "Genotype": [genotypes[0]] * len(group1) + [genotypes[1]] * len(group2),
            "Pretraining": [pretraining_vals[0]] * len(group1) + [pretraining_vals[1]] * len(group2),
            metric: np.concatenate([group1, group2]),
        }
    )

    # Generate visualizations
    plot_power_analysis(power_results, effect_size, output_dir)

    # Generate report
    generate_report(report_data, metric, power_results, effect_size, ci_lower, ci_upper, output_dir)

    print("\n" + "=" * 80)
    print("✅ Power analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Power analysis for TNT MB247 distance_moved metric with multiple comparisons"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--desired-power", type=float, default=0.8, help="Desired statistical power (default: 0.8)")
    parser.add_argument(
        "--correction",
        type=str,
        default="bonferroni",
        choices=["bonferroni", "none"],
        help="Multiple testing correction method (default: bonferroni)",
    )

    args = parser.parse_args()

    main(alpha=args.alpha, desired_power=args.desired_power, correction=args.correction)
