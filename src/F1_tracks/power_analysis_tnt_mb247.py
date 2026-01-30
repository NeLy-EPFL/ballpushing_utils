#!/usr/bin/env python3
"""
Power analysis for F1 TNT genotype comparison with multiple comparisons.

This script performs statistical power analysis to determine how many additional
samples are needed to detect a significant pretraining effect (if one exists)
for each genotype in the dataset.

The script analyzes:
- Within each genotype: Compare pretrained (y) vs naive (n)

This matches the comparison strategy used in f1_tnt_genotype_comparison.py.
Multiple testing correction (Bonferroni or FDR) is applied to adjust the significance level.

Usage:
    python power_analysis_tnt_mb247.py [--alpha 0.05] [--desired-power 0.8] [--correction fdr]
    python power_analysis_tnt_mb247.py --metric ball_proximity_proportion_200px
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.regression.mixed_linear_model import MixedLM
from f1_pca_analysis import PCA_METRICS

# Suppress convergence warnings from statsmodels during power simulations
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Hessian matrix.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# GENOTYPE SELECTION - Should match f1_tnt_genotype_comparison.py
# ============================================================================
SELECTED_GENOTYPES = [
    # === CONTROLS ===
    "TNTxEmptyGal4",
    "TNTxEmptySplit",
    #'TNTxPR',
    # === EXPERIMENTAL GENOTYPES (paired with internal controls) ===
    "TNTxDDC",
    "TNTxTH",
    #'PRxTH',
    "TNTxTRH",
    "TNTxMB247",
    #'PRxMB247',
    "TNTxLC10-2",
    "TNTxLC16-1",
    #'PRxLC16-1',
]


def parse_fly_name(fly_name):
    """
    Extract arena number and side from fly name.

    Format: 260114_TNT_F1_Videos_Checked_arena1_Left

    Parameters
    ----------
    fly_name : str
        Name of the fly

    Returns
    -------
    tuple
        (arena_number, arena_side)
    """
    parts = str(fly_name).split("_")
    arena_num = None
    side = None

    for part in parts:
        if "arena" in part.lower():
            # Extract arena number (e.g., "arena1" -> "1")
            arena_num = part.lower().replace("arena", "")
        if part in ["Left", "Right"]:
            side = part

    return arena_num, side


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


def power_analysis_lmm(group1_data, group2_data, alpha=0.05, desired_power=0.8, n_simulations=500):
    """
    Perform power analysis for LMM with random effects using simulation.

    This simulates increasing sample sizes and calculates the power to detect
    the observed effect using Linear Mixed-Effects Models with random effects
    for fly, date, arena, etc.

    Parameters
    ----------
    group1_data : pd.DataFrame
        Data for group 1 (naive), must have: metric, fly, pretraining, date, arena
    group2_data : pd.DataFrame
        Data for group 2 (pretrained), must have: metric, fly, pretraining, date, arena
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8)
    n_simulations : int
        Number of simulations per sample size (default: 500, reduced for LMM computational cost)

    Returns
    -------
    dict
        Dictionary with power analysis results
    """
    # Get current statistics
    current_n1 = len(group1_data)
    current_n2 = len(group2_data)
    metric_col = group1_data.columns[0]  # First column is the metric

    # Test range of sample sizes (from current to 5x current)
    max_multiplier = 5
    sample_sizes = list(range(current_n1, current_n1 * max_multiplier, max(1, current_n1 // 10)))

    if not sample_sizes:
        sample_sizes = [current_n1]

    powers = []

    print(f"\n🔬 Running LMM-based power analysis simulations...")
    print(f"   Current sample sizes: Group1={current_n1}, Group2={current_n2}")
    print(f"   Testing sample sizes from {min(sample_sizes)} to {max(sample_sizes)}")
    print(f"   Using {n_simulations} simulations per sample size (LMM with random effects)")

    for n in sample_sizes:
        n_significant = 0

        # Simulate n_simulations experiments with sample size n
        for sim_idx in range(n_simulations):
            try:
                # Sample with replacement to create datasets of size n
                idx1 = np.random.choice(group1_data.index, size=n, replace=True)
                idx2 = np.random.choice(group2_data.index, size=n, replace=True)

                sim_group1 = group1_data.loc[idx1].copy()
                sim_group2 = group2_data.loc[idx2].copy()

                # Combine into one dataset with pretraining indicator
                sim_data = pd.concat([sim_group1, sim_group2], ignore_index=True)

                # Ensure required columns exist
                if "fly" not in sim_data.columns:
                    sim_data["fly"] = [f"fly_{i}" for i in range(len(sim_data))]
                if "date" not in sim_data.columns and "Date" not in sim_data.columns:
                    sim_data["date"] = "unknown"
                if "arena" not in sim_data.columns:
                    sim_data["arena"] = "unknown"

                # Parse fly name to extract arena info if present
                if sim_data["fly"].notna().any():
                    arena_info = sim_data["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))
                    sim_data[["arena_number", "arena_side"]] = arena_info
                else:
                    sim_data["arena_number"] = None
                    sim_data["arena_side"] = None

                # Fit LMM
                formula = f"{metric_col} ~ pretraining_numeric"
                groups = sim_data["fly"]

                if len(groups.unique()) > 1:
                    try:
                        # Try primary optimizer (lbfgs)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = MixedLM.from_formula(formula, data=sim_data, groups=groups, re_formula="1").fit(
                                method="lbfgs", disp=False
                            )
                        p_value = model.pvalues.get("pretraining_numeric", 1.0)
                    except:
                        try:
                            # Try alternative optimizer (powell) if lbfgs fails
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = MixedLM.from_formula(formula, data=sim_data, groups=groups, re_formula="1").fit(
                                    method="powell", disp=False
                                )
                            p_value = model.pvalues.get("pretraining_numeric", 1.0)
                        except:
                            # Fall back to OLS if both LMM optimizers fail
                            model = smf.ols(formula, data=sim_data).fit()
                            p_value = model.pvalues.get("pretraining_numeric", 1.0)
                else:
                    # Not enough flies for random effects, use OLS
                    model = smf.ols(formula, data=sim_data).fit()
                    p_value = model.pvalues.get("pretraining_numeric", 1.0)

                if p_value < alpha:
                    n_significant += 1
            except Exception as e:
                # Skip this simulation if it fails
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


def power_analysis_bootstrap_controls(
    control_naive_data, control_pretrained_data, alpha=0.05, desired_power=0.8, n_bootstrap=1000
):
    """
    Perform power analysis using bootstrap sampling from control data.

    This approach:
    1. Uses actual control flies (pooled TNTxEmpty* genotypes)
    2. Bootstraps random samples of size n from naive and pretrained controls
    3. Tests power to detect the actual pretraining effect in controls
    4. More empirically grounded than assuming a pre-specified effect size

    Parameters
    ----------
    control_naive_data : pd.Series or array
        Metric values for control flies without pretraining
    control_pretrained_data : pd.Series or array
        Metric values for control flies with pretraining
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8)
    n_bootstrap : int
        Number of bootstrap iterations per sample size (default: 1000)

    Returns
    -------
    dict
        Dictionary with power analysis results
    """
    # Convert to arrays
    naive_vals = np.array(control_naive_data).flatten()
    pretrained_vals = np.array(control_pretrained_data).flatten()

    # Calculate observed effect size in controls
    observed_d = calculate_effect_size_cohens_d(naive_vals, pretrained_vals)

    print(f"\n   Control groups effect size (observed): Cohen's d = {observed_d:.3f}")
    print(
        f"   Control naive n = {len(naive_vals)}, mean = {np.mean(naive_vals):.2f} ± {np.std(naive_vals, ddof=1):.2f}"
    )
    print(
        f"   Control pretrained n = {len(pretrained_vals)}, mean = {np.mean(pretrained_vals):.2f} ± {np.std(pretrained_vals, ddof=1):.2f}"
    )

    # Test range of sample sizes
    min_n = min(5, len(naive_vals) // 2)
    max_n = len(naive_vals)
    sample_sizes = list(range(min_n, max_n + 1, max(1, (max_n - min_n) // 10)))
    if not sample_sizes or sample_sizes[-1] < max_n:
        sample_sizes.append(max_n)

    powers = []

    print(f"\n   Bootstrap power analysis ({n_bootstrap} resamples per sample size)...")
    print(f"   Testing sample sizes from {min(sample_sizes)} to {max(sample_sizes)}")

    for n in sample_sizes:
        n_significant = 0

        # Bootstrap n_bootstrap iterations
        for _ in range(n_bootstrap):
            # Sample n flies from naive and pretrained controls
            sample_naive = np.random.choice(naive_vals, size=n, replace=True)
            sample_pretrained = np.random.choice(pretrained_vals, size=n, replace=True)

            # Simple t-test
            try:
                from scipy.stats import ttest_ind

                _, p_value = ttest_ind(sample_naive, sample_pretrained)
                if p_value < alpha:
                    n_significant += 1
            except:
                pass

        power = n_significant / n_bootstrap
        powers.append(power)

        if power >= desired_power and len([p for p in powers if p >= desired_power]) == 1:
            print(f"   ✓ Desired power ({desired_power:.0%}) first achieved at n={n}")

    # Find target sample size
    target_n = None
    for n, power in zip(sample_sizes, powers):
        if power >= desired_power:
            target_n = n
            break

    return {
        "sample_sizes": sample_sizes,
        "powers": powers,
        "observed_effect_size": observed_d,
        "control_naive_n": len(naive_vals),
        "control_pretrained_n": len(pretrained_vals),
        "target_n": target_n,
        "desired_power": desired_power,
        "alpha": alpha,
    }


def power_analysis_lmm_with_effect(
    group1_data, group2_data, effect_size_cohens_d=0.5, alpha=0.05, desired_power=0.8, n_simulations=500
):
    """
    Perform power analysis for LMM assuming a specified effect size.

    This approach:
    1. Takes actual data from both groups (but ignores observed differences)
    2. Simulates group 2 with a SPECIFIED effect size relative to group 1
    3. Tests power to detect that pre-specified effect size

    This avoids p-hacking by not tailoring the effect size to post-hoc observations.

    Parameters
    ----------
    group1_data : pd.DataFrame
        Data for group 1 (baseline/control)
    group2_data : pd.DataFrame
        Data for group 2 (will add specified effect size in simulations)
    effect_size_cohens_d : float
        Target Cohen's d effect size to detect (default: 0.5 = "medium")
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8)
    n_simulations : int
        Number of simulations per sample size (default: 500)

    Returns
    -------
    dict
        Dictionary with power analysis results
    """
    current_n1 = len(group1_data)
    current_n2 = len(group2_data)

    # Find the metric column (first numeric column)
    metric_col = None
    for col in group1_data.columns:
        if pd.api.types.is_numeric_dtype(group1_data[col]):
            metric_col = col
            break

    if metric_col is None:
        raise ValueError("No numeric column found in data for metric")

    # Calculate pooled SD to convert Cohen's d to absolute effect
    all_values = pd.concat([group1_data[metric_col], group2_data[metric_col]])
    pooled_sd = np.std(all_values, ddof=1)
    absolute_effect = effect_size_cohens_d * pooled_sd

    # Test range of sample sizes (from current to 5x current)
    max_multiplier = 5
    sample_sizes = list(range(current_n1, current_n1 * max_multiplier, max(1, current_n1 // 10)))
    if not sample_sizes:
        sample_sizes = [current_n1]

    powers = []

    print(f"\n   Pooled SD: {pooled_sd:.4f}")
    print(f"   Target effect size (Cohen's d): {effect_size_cohens_d}")
    print(f"   Absolute effect to detect: {absolute_effect:.4f}")
    print(f"\n   Testing sample sizes from {min(sample_sizes)} to {max(sample_sizes)}")

    for n in sample_sizes:
        n_significant = 0

        # Simulate n_simulations experiments with sample size n
        for sim_idx in range(n_simulations):
            try:
                # Sample with replacement to create datasets of size n
                idx1 = np.random.choice(group1_data.index, size=n, replace=True)
                idx2 = np.random.choice(group2_data.index, size=n, replace=True)

                sim_group1 = group1_data.loc[idx1].copy()
                sim_group2 = group2_data.loc[idx2].copy()

                # Add the specified effect size to group 2
                sim_group2[metric_col] = sim_group2[metric_col] + absolute_effect

                # Combine into one dataset with pretraining indicator
                sim_data = pd.concat([sim_group1, sim_group2], ignore_index=True)

                # Ensure required columns exist
                if "fly" not in sim_data.columns:
                    sim_data["fly"] = [f"fly_{i}" for i in range(len(sim_data))]
                if "date" not in sim_data.columns and "Date" not in sim_data.columns:
                    sim_data["date"] = "unknown"
                if "arena" not in sim_data.columns:
                    sim_data["arena"] = "unknown"

                # Parse fly name to extract arena info if present
                if sim_data["fly"].notna().any():
                    arena_info = sim_data["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))
                    sim_data[["arena_number", "arena_side"]] = arena_info
                else:
                    sim_data["arena_number"] = None
                    sim_data["arena_side"] = None

                # Fit LMM
                formula = f"{metric_col} ~ pretraining_numeric"
                groups = sim_data["fly"]

                if len(groups.unique()) > 1:
                    try:
                        # Try primary optimizer (lbfgs)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = MixedLM.from_formula(formula, data=sim_data, groups=groups, re_formula="1").fit(
                                method="lbfgs", disp=False
                            )
                        p_value = model.pvalues.get("pretraining_numeric", 1.0)
                    except:
                        try:
                            # Try alternative optimizer (powell) if lbfgs fails
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = MixedLM.from_formula(formula, data=sim_data, groups=groups, re_formula="1").fit(
                                    method="powell", disp=False
                                )
                            p_value = model.pvalues.get("pretraining_numeric", 1.0)
                        except:
                            # Fall back to OLS if both LMM optimizers fail
                            model = smf.ols(formula, data=sim_data).fit()
                            p_value = model.pvalues.get("pretraining_numeric", 1.0)
                else:
                    # Not enough flies for random effects, use OLS
                    model = smf.ols(formula, data=sim_data).fit()
                    p_value = model.pvalues.get("pretraining_numeric", 1.0)

                if p_value < alpha:
                    n_significant += 1
            except Exception as e:
                # Skip this simulation if it fails
                pass

        power = n_significant / n_simulations
        powers.append(power)

        # Print progress for key milestones
        if power >= desired_power and len([p for p in powers if p >= desired_power]) == 1:
            print(f"   ✓ Desired power ({desired_power:.0%}) first achieved at n={n}")

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
        "effect_size": effect_size_cohens_d,
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
        f"Power Analysis for LMM (Linear Model)\nEffect Size (Cohen's d) = {effect_size:.3f} ({interpret_cohens_d(effect_size)})",
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
    report_lines.append(f"POWER ANALYSIS REPORT: F1 TNT - {metric} (Within-Genotype Comparisons)")
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

    # All comparisons are within-genotype
    report_lines.append("## COMPARISONS ANALYZED")
    report_lines.append("-" * 100)
    report_lines.append("All comparisons are within-genotype (Pretrained vs Naive)")
    report_lines.append(f"Total comparisons: {len(all_results)}")
    report_lines.append("")
    for r in all_results:
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


def analyze_single_metric(metric, alpha=0.05, desired_power=0.8, verbose=True):
    """
    Run bootstrap power analysis for a single metric.

    Parameters
    ----------
    metric : str
        Metric to analyze
    alpha : float
        Significance level (default: 0.05)
    desired_power : float
        Target statistical power (default: 0.8 = 80%)
    verbose : bool
        Whether to print detailed output (default: True)

    Returns
    -------
    dict
        Results dictionary with metric name, current_n, target_n, observed_d, etc.
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"F1 TNT POWER ANALYSIS: {metric}")
        print("=" * 80)
        print(f"\nAnalysis Strategy: Bootstrap from control data")
        print(f"  • Pools control genotypes (TNTxEmpty*) together")
        print(f"  • Uses observed pretraining effect from controls")
        print(f"  • Bootstrap resampling to estimate power empirically")
        print(f"  • Avoids p-hacking through pre-specified control-based approach")

    # Load data - use the full F1 TNT dataset
    dataset_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260119_10_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather"
    )

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    print(f"\n📁 Loading dataset from: {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

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
    # Find ball condition column (prefer ball_identity over ball_condition)
    ball_condition_col = None
    if "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"
    elif "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"

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

    # Filter for test ball only (matching f1_tnt_genotype_comparison.py)
    if ball_condition_col:
        test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
        initial_size = len(df)

        # Filter for test ball
        mask = df[ball_condition_col].isin(test_ball_values)
        df = df[mask].copy()

        if len(df) > 0:
            print(f"✓ Filtered for test ball: {initial_size} → {len(df)} rows")
        else:
            print(f"⚠️  Warning: No test ball data found")
            print(f"   Ball condition values in data: {df[ball_condition_col].unique()}")
            print(f"   Proceeding with all data...")
            df = pd.read_feather(dataset_path)  # Reload without filter

    # Clean data
    df_clean = df[[genotype_col, pretraining_col, metric]].dropna()
    df_clean = df_clean[~np.isinf(df_clean[metric])]

    print(f"✓ Clean data: {len(df_clean)} rows")

    # Get unique genotypes and pretraining values
    genotypes = sorted(df_clean[genotype_col].unique())
    pretraining_vals = sorted(df_clean[pretraining_col].unique())

    print(f"\n🔬 All genotypes in dataset: {genotypes}")

    # Filter for selected genotypes (matching f1_tnt_genotype_comparison.py)
    if SELECTED_GENOTYPES:
        genotypes_in_data = [g for g in SELECTED_GENOTYPES if g in genotypes]
        if genotypes_in_data:
            genotypes = genotypes_in_data
            df_clean = df_clean[df_clean[genotype_col].isin(SELECTED_GENOTYPES)]
            print(f"✓ Filtered to {len(genotypes)} selected genotypes: {genotypes}")
        else:
            print(f"⚠️  Warning: None of the selected genotypes found in data")
            print(f"   Will use all genotypes in dataset")

    print(f"   Pretraining conditions: {pretraining_vals}")

    # Identify control genotypes (those with "Empty" in the name)
    control_genotypes = [g for g in genotypes if "Empty" in g]
    experimental_genotypes = [g for g in genotypes if "Empty" not in g]

    # ========================================================================
    # PRINCIPLED POWER ANALYSIS: Based on overall data variability
    # ========================================================================

    # ========================================================================
    # PRINCIPLED POWER ANALYSIS: Bootstrap from control data
    # ========================================================================

    if verbose:
        print(f"\n📊 STEP 1: Extract control group data")

    # Bootstrap configuration
    n_bootstrap = 1000

    # Get the full original dataset filtered to selected genotypes
    df_analysis = df[df[genotype_col].isin(SELECTED_GENOTYPES)].copy()
    df_analysis = df_analysis.dropna(subset=[metric])
    df_analysis = df_analysis[~np.isinf(df_analysis[metric])]

    # Identify control genotypes
    control_mask = df_analysis[genotype_col].str.contains("Empty", na=False)
    df_controls = df_analysis[control_mask].copy()

    # Aggregate to fly level
    if "fly" in df_controls.columns:
        fly_controls = (
            df_controls.groupby("fly", as_index=False)
            .agg(
                {
                    metric: "mean",
                    pretraining_col: "first",  # All observations for a fly have same pretraining status
                }
            )
            .copy()
        )
    else:
        fly_controls = df_controls[[metric, pretraining_col]].copy()

    # Split by pretraining status
    naive_controls = fly_controls[fly_controls[pretraining_col] == "n"][metric].values
    pretrained_controls = fly_controls[fly_controls[pretraining_col] == "y"][metric].values

    # Calculate per-genotype sample sizes (for realistic reporting)
    n_control_genotypes = df_controls[genotype_col].nunique()
    avg_n_per_genotype = (
        min(len(naive_controls), len(pretrained_controls)) / n_control_genotypes if n_control_genotypes > 0 else 0
    )

    if verbose:
        print(f"   Control flies (pooled TNTxEmpty*):")
        print(f"   - Naive (n): {len(naive_controls)} flies")
        print(f"   - Pretrained (y): {len(pretrained_controls)} flies")
        print(f"   - Number of control genotypes pooled: {n_control_genotypes}")
        print(f"   - Average n per individual genotype: {avg_n_per_genotype:.1f}")

    if len(naive_controls) < 5 or len(pretrained_controls) < 5:
        if verbose:
            print(f"\n⚠️  Warning: Not enough control flies for power analysis")
            print(f"   Need at least 5 per condition, found {len(naive_controls)} and {len(pretrained_controls)}")
        return None

    if verbose:
        print(f"\n📊 STEP 2: Run bootstrap power analysis")
        print(f"   Using actual pretraining effect from control groups")
        print(f"   Resampling {n_bootstrap} times per sample size")

    # Run power analysis with bootstrap
    power_results = power_analysis_bootstrap_controls(
        naive_controls,
        pretrained_controls,
        alpha=alpha,
        desired_power=desired_power,
        n_bootstrap=n_bootstrap,
    )

    # ========================================================================
    # RESULTS AND RECOMMENDATIONS
    # ========================================================================

    if verbose:
        print(f"\n" + "=" * 80)
        print("RESULTS AND RECOMMENDATIONS")
        print("=" * 80)

    # Use per-genotype average as "current n" (more realistic for planning)
    current_n = int(avg_n_per_genotype)
    target_n = power_results["target_n"]
    observed_d = power_results["observed_effect_size"]

    # Get power at current sample sizes
    sample_sizes = power_results["sample_sizes"]
    powers = power_results["powers"]

    # Find power at current_n (interpolate if needed)
    if current_n in sample_sizes:
        idx = sample_sizes.index(current_n)
        current_power = powers[idx]
    else:
        # Interpolate
        current_power = np.interp(current_n, sample_sizes, powers)

    if verbose:
        print(f"\n📈 Current Control Data Analysis:")
        print(f"   Current n per individual genotype: {current_n}")
        print(f"   Observed effect size (Cohen's d): {observed_d:.3f} ({interpret_cohens_d(observed_d)})")
        print(f"   Current statistical power: {current_power:.1%}")

        if target_n:
            additional = target_n - current_n
            print(f"\n🎯 To achieve {desired_power:.0%} power:")
            print(f"   Recommended n per group: {target_n}")
            print(f"   Additional samples needed: {additional}")
            if additional > 0:
                print(f"   Total increase: {(additional / current_n * 100):.0f}% more samples")
            else:
                print(f"   ✓ You already have sufficient power!")
        else:
            print(f"\n⚠️  Could not achieve {desired_power:.0%} power within tested range")
            max_n = sample_sizes[-1]
            max_power = powers[-1]
            print(f"   Maximum tested: n={max_n} gave power={max_power:.1%}")
    # Return results dictionary
    results = {
        "metric": metric,
        "current_n": current_n,
        "target_n": target_n,
        "observed_d": observed_d,
        "current_power": current_power,
        "additional_needed": target_n - current_n if target_n else None,
        "pooled_naive_n": len(naive_controls),
        "pooled_pretrained_n": len(pretrained_controls),
        "n_control_genotypes": n_control_genotypes,
    }

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    if verbose:
        print(f"\n📊 Creating visualizations...")

    if not verbose:
        # Skip visualization if not verbose
        return results

    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/power_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create power analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Power curve
    ax1 = axes[0]
    sample_sizes = power_results["sample_sizes"]
    powers = power_results["powers"]
    # current_n already calculated above
    # target_n already calculated above

    ax1.plot(sample_sizes, powers, "b-", linewidth=2.5, label="Statistical Power", marker="o", markersize=6)
    ax1.axhline(y=desired_power, color="r", linestyle="--", linewidth=2, label=f"Desired Power ({desired_power:.0%})")
    ax1.axvline(x=current_n, color="green", linestyle="--", linewidth=2, label=f"Current n={current_n}")

    if target_n:
        ax1.axvline(x=target_n, color="orange", linestyle="--", linewidth=2, label=f"Target n={target_n}")
        ax1.plot(target_n, desired_power, "r*", markersize=15)

    ax1.set_xlabel("Sample Size per Group", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Statistical Power", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Power Analysis (Bootstrap from Controls)\nObserved Effect Size (Cohen's d) = {observed_d:.3f} ({interpret_cohens_d(observed_d)})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Sample size requirements for different power levels
    ax2 = axes[1]
    power_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    required_ns = []

    for power_level in power_levels:
        n_needed = None
        for n, power in zip(sample_sizes, powers):
            if power >= power_level:
                n_needed = n
                break
        if n_needed is None:
            n_needed = sample_sizes[-1]  # Use max if not reached
        required_ns.append(n_needed)

    bars = ax2.bar(
        range(len(power_levels)), required_ns, color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax2.axhline(y=current_n, color="green", linestyle="--", linewidth=2, label=f"Current n={current_n}")
    ax2.set_xticks(range(len(power_levels)))
    ax2.set_xticklabels([f"{p:.0%}" for p in power_levels], fontsize=10)
    ax2.set_xlabel("Desired Statistical Power", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Required Sample Size per Group", fontsize=12, fontweight="bold")
    ax2.set_title("Sample Size Requirements\nfor Different Power Levels", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, n in zip(bars, required_ns):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plot_path = output_dir / "power_analysis_principled.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Power plot saved to: {plot_path}")
    plt.close()

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"PRINCIPLED POWER ANALYSIS REPORT: {metric}")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("ANALYSIS APPROACH")
    report_lines.append("-" * 80)
    report_lines.append("This analysis uses a BOOTSTRAP approach from control data:")
    report_lines.append("  • Pools control flies (TNTxEmptyGal4 + TNTxEmptySplit)")
    report_lines.append("  • Bootstraps random samples from actual pretraining effect in controls")
    report_lines.append("  • Empirically estimates power based on real data variability")
    report_lines.append("  • No assumptions about effect sizes - uses observed control effect")
    report_lines.append("")

    report_lines.append("DATA SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Metric analyzed: {metric}")
    report_lines.append(f"Control flies (pooled TNTxEmpty*):")
    report_lines.append(
        f"  - Naive (n): {len(naive_controls)} flies, mean = {np.mean(naive_controls):.2f} ± {np.std(naive_controls, ddof=1):.2f}"
    )
    report_lines.append(
        f"  - Pretrained (y): {len(pretrained_controls)} flies, mean = {np.mean(pretrained_controls):.2f} ± {np.std(pretrained_controls, ddof=1):.2f}"
    )
    report_lines.append("")

    report_lines.append("STATISTICAL PARAMETERS")
    report_lines.append("-" * 80)
    report_lines.append(f"Observed pretraining effect (Cohen's d): {observed_d:.3f}")
    report_lines.append(f"Interpretation: {interpret_cohens_d(observed_d).capitalize()}")
    report_lines.append(f"Significance level (α): {alpha}")
    report_lines.append(f"Desired power: {desired_power:.0%}")
    report_lines.append(f"Bootstrap resampling: {n_bootstrap} iterations per sample size")
    report_lines.append("")

    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append(f"Current sample size per group: {current_n}")
    report_lines.append(f"Current statistical power (at n={current_n}): {current_power:.1%}")
    report_lines.append("")

    if target_n:
        additional = target_n - current_n
        report_lines.append(f"To achieve {desired_power:.0%} power to detect the observed pretraining effect:")
        report_lines.append(f"  • Recommended sample size per group: {target_n}")
        report_lines.append(f"  • Additional samples needed: {additional}")
        if additional > 0:
            report_lines.append(f"  • Total increase: {(additional / current_n * 100):.0f}%")
        else:
            report_lines.append(f"  • Status: ✓ Already sufficient power!")
    else:
        max_n = sample_sizes[-1]
        max_power = powers[-1]
        report_lines.append(f"⚠️  Could not achieve {desired_power:.0%} power within tested range (n up to {max_n})")
        report_lines.append(f"  • Maximum power achieved: {max_power:.1%} at n={max_n}")

    report_lines.append("")
    report_lines.append("EFFECT SIZE INTERPRETATION GUIDE")
    report_lines.append("-" * 80)
    report_lines.append("  • Negligible: |d| < 0.2")
    report_lines.append("  • Small:      0.2 ≤ |d| < 0.5")
    report_lines.append("  • Medium:     0.5 ≤ |d| < 0.8")
    report_lines.append("  • Large:      |d| ≥ 0.8")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print and save report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / "power_analysis_principled_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n✓ Report saved to: {report_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("✅ Bootstrap power analysis complete!")
        print("=" * 80)

    return results


def main(alpha=0.05, desired_power=0.8, metrics=None):
    """
    Run power analysis across multiple metrics.

    Parameters
    ----------
    alpha : float
        Significance level
    desired_power : float
        Target statistical power
    metrics : list or None
        List of metrics to analyze (default: all PCA_METRICS)
    """
    if metrics is None:
        metrics = PCA_METRICS

    print("\n" + "=" * 80)
    print("F1 TNT POWER ANALYSIS - MULTI-METRIC SUMMARY")
    print("=" * 80)
    print(f"\nAnalyzing {len(metrics)} metrics using bootstrap from control data")
    print(f"Alpha: {alpha}, Desired power: {desired_power:.0%}\n")

    all_results = []
    for i, metric in enumerate(metrics, 1):
        print(f"\n[{i}/{len(metrics)}] Analyzing: {metric}")
        print("-" * 80)

        result = analyze_single_metric(metric, alpha=alpha, desired_power=desired_power, verbose=False)
        if result:
            all_results.append(result)
            print(
                f"  Current n/genotype: {result['current_n']}, Target n: {result['target_n']}, "
                f"Observed d: {result['observed_d']:.3f}, Current power: {result['current_power']:.1%}"
            )
        else:
            print(f"  ⚠️  Insufficient data")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    summary_df = pd.DataFrame(all_results)
    if len(summary_df) > 0:
        # Round for display
        display_df = summary_df[
            ["metric", "current_n", "target_n", "observed_d", "current_power", "additional_needed"]
        ].copy()
        display_df["observed_d"] = display_df["observed_d"].round(3)
        display_df["current_power"] = (display_df["current_power"] * 100).round(1)
        display_df.columns = ["Metric", "Current N", "Target N", "Cohen's d", "Power (%)", "Additional Needed"]

        print("\n" + display_df.to_string(index=False))

        # Overall recommendation
        max_target = display_df["Target N"].max()
        print(f"\n" + "=" * 80)
        print(f"OVERALL RECOMMENDATION")
        print("=" * 80)
        print(f"\nTo achieve {desired_power:.0%} power across ALL metrics:")
        print(f"  Recommended n per genotype: {int(max_target)}")
        print(f"  (Highest requirement across all {len(metrics)} metrics)")

        # Save summary
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/power_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "power_analysis_summary_all_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ Multi-metric power analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap power analysis for F1 TNT experiments using control data")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--desired-power", type=float, default=0.8, help="Desired statistical power (default: 0.8)")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Single metric to analyze (default: analyze all PCA metrics)",
    )

    args = parser.parse_args()

    if args.metric:
        # Single metric mode with full verbose output
        result = analyze_single_metric(args.metric, alpha=args.alpha, desired_power=args.desired_power, verbose=True)
    else:
        # Multi-metric mode
        main(alpha=args.alpha, desired_power=args.desired_power)
