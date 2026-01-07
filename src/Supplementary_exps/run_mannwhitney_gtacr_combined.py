#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for Gtacr OR67d experiments
with combined ATR/no-ATR visualization.

This script generates comprehensive visualizations showing all genotypes with both
ATR conditions (yes/no) on the same plot:
- 2-level grouping: Genotype + ATR condition
- Dashed boxplots for ATR='no', solid for ATR='yes'
- Comparison across all Genotype+ATR groups
- FDR correction across all comparisons for each metric
- Significance-based sorting
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)

Usage:
    python run_mannwhitney_gtacr_combined.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: If specified, skip metrics that already have plots.
    --test: If specified, only process first 3 metrics for debugging.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def generate_combined_mannwhitney_plots(
    data,
    metrics,
    genotype_col="Genotype",
    atr_col="ATR",
    control_genotype="GtacrxOR67d",
    control_atr="yes",
    figsize=(18, 10),
    output_dir="mann_whitney_plots_combined",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with both ATR conditions shown together.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        genotype_col (str): Column name for genotype. Default is "Genotype".
        atr_col (str): Column name for ATR condition. Default is "ATR".
        control_genotype (str): Name of control genotype. Default is "GtacrxOR67d".
        control_atr (str): ATR condition for control. Default is "yes".
        figsize (tuple): Size of each figure. Default is (18, 10).
        output_dir: Directory to save the plots.
        fdr_method (str): Method for FDR correction. Default is "fdr_bh".
        alpha (float): Significance level after FDR correction. Default is 0.05.

    Returns:
        pd.DataFrame: Statistics table with Mann-Whitney U test results and FDR correction.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    # Create combined grouping variable
    data["Genotype_ATR"] = data[genotype_col] + " (ATR=" + data[atr_col].astype(str) + ")"

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating combined Mann-Whitney plot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        # Clean metric data - handle string representations and ensure numeric
        metric_data = data[metric].copy()

        # Check if metric contains string representations of lists or non-numeric data
        if metric_data.dtype == 'object':
            # Try to convert strings to numeric, set non-convertible to NaN
            metric_data = pd.to_numeric(metric_data, errors='coerce')
            if metric_data.isna().all():
                print(f"  ⚠️  Warning: Metric '{metric}' contains non-numeric data. Skipping.")
                continue

        # Update the metric column in data
        data[metric] = metric_data

        plot_data = data.dropna(subset=[metric, genotype_col, atr_col]).copy()

        # Define control group
        control_group = f"{control_genotype} (ATR={control_atr})"

        # Check if control exists
        if control_group not in plot_data["Genotype_ATR"].unique():
            print(f"  ⚠️  Warning: Control group '{control_group}' not found. Skipping {metric}.")
            continue

        # Get control data
        control_vals = plot_data[plot_data["Genotype_ATR"] == control_group][metric].dropna()
        if len(control_vals) < 2:
            print(f"  ⚠️  Warning: Insufficient control data for {metric}. Skipping.")
            continue

        control_median = control_vals.median()

        # Get all unique groups
        all_groups = sorted(plot_data["Genotype_ATR"].unique())
        test_groups = [g for g in all_groups if g != control_group]

        # ============================================================
        # PRIMARY TEST: Control vs ALL OTHER groups pooled
        # ============================================================
        pooled_test_vals = plot_data[plot_data["Genotype_ATR"] != control_group][metric].dropna()

        pooled_stats = None
        if len(pooled_test_vals) >= 2:
            try:
                pooled_statistic, pooled_pval = mannwhitneyu(control_vals, pooled_test_vals, alternative="two-sided")
                pooled_median = pooled_test_vals.median()
                pooled_effect_size = pooled_median - control_median

                if pooled_median > control_median:
                    pooled_direction = "increased"
                elif pooled_median < control_median:
                    pooled_direction = "decreased"
                else:
                    pooled_direction = "neutral"

                pooled_stats = {
                    "group": "ALL_OTHER_GROUPS_POOLED",
                    "genotype": "Pooled",
                    "atr": "mixed",
                    "n_control": len(control_vals),
                    "n_test": len(pooled_test_vals),
                    "control_median": control_median,
                    "test_median": pooled_median,
                    "effect_size": pooled_effect_size,
                    "statistic": pooled_statistic,
                    "pval": pooled_pval,
                    "pval_corrected": pooled_pval,  # No correction needed for single test
                    "significant": pooled_pval < alpha,
                    "direction": pooled_direction,
                    "test_type": "primary_pooled",
                }

                # Significance stars for pooled test
                if pooled_pval < 0.001:
                    pooled_stats["sig_stars"] = "***"
                elif pooled_pval < 0.01:
                    pooled_stats["sig_stars"] = "**"
                elif pooled_pval < 0.05:
                    pooled_stats["sig_stars"] = "*"
                else:
                    pooled_stats["sig_stars"] = "ns"

                print(f"  PRIMARY TEST: Control vs ALL OTHERS pooled: p={pooled_pval:.3e} {pooled_stats['sig_stars']}")

            except Exception as e:
                print(f"  ⚠️  Error in pooled comparison: {e}")

        # ============================================================
        # SECONDARY TESTS: Individual pairwise comparisons
        # ============================================================
        stats_list = []
        p_values = []

        for test_group in test_groups:
            test_vals = plot_data[plot_data["Genotype_ATR"] == test_group][metric].dropna()

            if len(test_vals) < 2:
                continue

            # Mann-Whitney U test
            try:
                statistic, pval = mannwhitneyu(control_vals, test_vals, alternative="two-sided")
            except Exception as e:
                print(f"  ⚠️  Error in Mann-Whitney test for {test_group}: {e}")
                continue

            test_median = test_vals.median()
            effect_size = test_median - control_median

            # Determine direction
            if test_median > control_median:
                direction = "increased"
            elif test_median < control_median:
                direction = "decreased"
            else:
                direction = "neutral"

            stats_list.append({
                "group": test_group,
                "genotype": test_group.split(" (ATR=")[0],
                "atr": test_group.split("ATR=")[1].rstrip(")"),
                "n_control": len(control_vals),
                "n_test": len(test_vals),
                "control_median": control_median,
                "test_median": test_median,
                "effect_size": effect_size,
                "statistic": statistic,
                "pval": pval,
                "direction": direction,
                "test_type": "pairwise",
            })
            p_values.append(pval)

        if len(p_values) == 0:
            print(f"  ⚠️  No valid pairwise comparisons for {metric}. Skipping.")
            if pooled_stats is None:
                continue

        # Apply FDR correction to pairwise comparisons
        if len(p_values) > 0:
            p_values = np.array(p_values)
            rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)
        else:
            rejected = np.array([])
            pvals_corrected = np.array([])

        # Add corrected p-values and significance to pairwise tests
        for i, stat_dict in enumerate(stats_list):
            stat_dict["pval_corrected"] = pvals_corrected[i]
            stat_dict["significant"] = rejected[i]

            # Significance stars (for pairwise comparisons with FDR correction)
            if pvals_corrected[i] < 0.001:
                stat_dict["sig_stars"] = "***"
            elif pvals_corrected[i] < 0.01:
                stat_dict["sig_stars"] = "**"
            elif pvals_corrected[i] < 0.05:
                stat_dict["sig_stars"] = "*"
            else:
                stat_dict["sig_stars"] = "ns"

        # Sort groups by significance and effect
        increased = [s for s in stats_list if s["significant"] and s["direction"] == "increased"]
        decreased = [s for s in stats_list if s["significant"] and s["direction"] == "decreased"]
        neutral = [s for s in stats_list if not s["significant"]]

        increased.sort(key=lambda x: x["test_median"], reverse=True)
        neutral.sort(key=lambda x: x["test_median"], reverse=True)
        decreased.sort(key=lambda x: x["test_median"])

        sorted_stats = increased + neutral + decreased
        sorted_groups = [control_group] + [s["group"] for s in sorted_stats]

        # Store results for output - PRIMARY TEST FIRST
        if pooled_stats:
            all_stats.append({
                "Metric": metric,
                "Control": control_group,
                "Test_Group": pooled_stats["group"],
                "Genotype": pooled_stats["genotype"],
                "ATR": pooled_stats["atr"],
                "n_control": pooled_stats["n_control"],
                "n_test": pooled_stats["n_test"],
                "Control_Median": pooled_stats["control_median"],
                "Test_Median": pooled_stats["test_median"],
                "Effect_Size": pooled_stats["effect_size"],
                "U_statistic": pooled_stats["statistic"],
                "p_value": pooled_stats["pval"],
                "p_value_corrected": pooled_stats["pval_corrected"],
                "significant": pooled_stats["significant"],
                "direction": pooled_stats["direction"],
                "sig_stars": pooled_stats["sig_stars"],
                "test_type": pooled_stats["test_type"],
            })

        # Store pairwise comparison results
        for stat_dict in stats_list:
            all_stats.append({
                "Metric": metric,
                "Control": control_group,
                "Test_Group": stat_dict["group"],
                "Genotype": stat_dict["genotype"],
                "ATR": stat_dict["atr"],
                "n_control": stat_dict["n_control"],
                "n_test": stat_dict["n_test"],
                "Control_Median": stat_dict["control_median"],
                "Test_Median": stat_dict["test_median"],
                "Effect_Size": stat_dict["effect_size"],
                "U_statistic": stat_dict["statistic"],
                "p_value": stat_dict["pval"],
                "p_value_corrected": stat_dict["pval_corrected"],
                "significant": stat_dict["significant"],
                "direction": stat_dict["direction"],
                "sig_stars": stat_dict["sig_stars"],
                "test_type": stat_dict["test_type"],
            })

        # ========================================
        # RESTRUCTURE: Group by Genotype, then by ATR within each genotype
        # ========================================

        # Get unique genotypes from the groups
        genotypes_in_data = sorted(set([g.split(" (ATR=")[0] for g in all_groups]))

        # Build hierarchical structure: for each genotype, list its ATR conditions
        hierarchical_groups = []
        genotype_labels = []  # For y-axis primary labels
        atr_labels = []  # For y-axis secondary labels

        for genotype in genotypes_in_data:
            # Find all ATR conditions for this genotype
            genotype_groups = [g for g in all_groups if g.startswith(genotype + " (ATR=")]

            # Sort by ATR: 'no' first (if exists), then 'yes'
            genotype_groups = sorted(genotype_groups, key=lambda x: ("yes" in x, x))

            for group in genotype_groups:
                hierarchical_groups.append(group)
                genotype_labels.append(genotype)
                # Extract ATR value
                atr_val = group.split("ATR=")[1].rstrip(")")
                atr_labels.append(f"ATR={atr_val}")

        # Update plot data with hierarchical ordering
        plot_data["Genotype_ATR"] = pd.Categorical(
            plot_data["Genotype_ATR"],
            categories=hierarchical_groups,
            ordered=True
        )
        plot_data = plot_data.sort_values(by=["Genotype_ATR"])

        # Calculate dynamic figure height
        n_groups = len(hierarchical_groups)
        fig_height = max(10, n_groups * 0.6)

        fig, ax = plt.subplots(figsize=(figsize[0], fig_height))

        # Determine line styles based on ATR condition
        box_styles = []
        for group in hierarchical_groups:
            if "(ATR=no)" in group or "(ATR=nan)" in group:
                box_styles.append("--")  # dashed for no ATR
            else:
                box_styles.append("-")   # solid for ATR

        # Create boxplots
        bp = ax.boxplot(
            [plot_data[plot_data["Genotype_ATR"] == group][metric].values for group in hierarchical_groups],
            positions=range(len(hierarchical_groups)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            vert=False,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color='black')
        )

        # Style boxes: no fill, black edges, dashed/solid based on ATR
        for patch, linestyle in zip(bp['boxes'], box_styles):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linestyle(linestyle)

        # Style whiskers and caps with matching line styles
        for i, linestyle in enumerate(box_styles):
            bp['whiskers'][i*2].set_linestyle(linestyle)
            bp['whiskers'][i*2+1].set_linestyle(linestyle)
            bp['caps'][i*2].set_linestyle(linestyle)
            bp['caps'][i*2+1].set_linestyle(linestyle)

        # Add scatter points: purple for OR67d genotypes, gray for others
        for i, group in enumerate(hierarchical_groups):
            group_data = plot_data[plot_data["Genotype_ATR"] == group][metric].values
            y_positions = np.random.normal(i, 0.04, size=len(group_data))

            # Color based on genotype
            if "OR67d" in group:
                color = '#9467bd'  # purple
            else:
                color = '#7f7f7f'  # gray

            ax.scatter(group_data, y_positions, alpha=0.4, s=20, color=color, zorder=3)

        # Add color-coded backgrounds based on significance
        sig_lookup = {}
        for result in sorted_stats:
            if result["significant"]:
                if result["direction"] == "increased":
                    sig_lookup[result["group"]] = "lightgreen"
                elif result["direction"] == "decreased":
                    sig_lookup[result["group"]] = "lightcoral"
                else:
                    sig_lookup[result["group"]] = "lightyellow"
            else:
                sig_lookup[result["group"]] = "lightyellow"

        # Get x-axis limits for rectangles
        x_min, x_max = ax.get_xlim()

        for i, group in enumerate(hierarchical_groups):
            if group == control_group:
                # Control group background
                rect = Rectangle(
                    (x_min, i - 0.5), x_max - x_min, 1,
                    facecolor='lightgray', alpha=0.3, zorder=0
                )
            else:
                # Test group background
                color = sig_lookup.get(group, "lightyellow")
                rect = Rectangle(
                    (x_min, i - 0.5), x_max - x_min, 1,
                    facecolor=color, alpha=0.3, zorder=0
                )
            ax.add_patch(rect)

        # Add separators between genotypes
        prev_genotype = None
        for i, genotype in enumerate(genotype_labels):
            if prev_genotype is not None and genotype != prev_genotype:
                # Add a horizontal line separator
                ax.axhline(y=i - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            prev_genotype = genotype

        # Add significance annotations
        x_max_data = plot_data[metric].max()
        x_range = plot_data[metric].max() - plot_data[metric].min()
        annotation_x = x_max_data + 0.05 * x_range
        pvalue_x = x_max_data + 0.18 * x_range  # Increased spacing for longer text

        # Add pooled test result at the top
        if pooled_stats:
            y_text_pos = len(hierarchical_groups) + 0.5
            pooled_text = f"PRIMARY: Control vs ALL OTHERS: {pooled_stats['sig_stars']} (p={pooled_stats['pval']:.3e})"
            ax.text(x_min, y_text_pos, pooled_text, ha='left', va='center',
                   fontsize=11, fontweight='bold',
                   color='darkgreen' if pooled_stats['significant'] else 'darkgray',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

        for i, group in enumerate(hierarchical_groups):
            if group == control_group:
                ax.text(annotation_x, i, "CTRL", ha='left', va='center',
                       fontsize=10, fontweight='bold')
            else:
                # Find stats for this group
                group_stats = next((s for s in sorted_stats if s["group"] == group), None)
                if group_stats:
                    sig_stars = group_stats["sig_stars"]
                    pval_corrected = group_stats["pval_corrected"]

                    # Stars (pairwise comparison)
                    ax.text(annotation_x, i, sig_stars, ha='left', va='center',
                           fontsize=11, fontweight='bold', color='darkblue')

                    # P-value (indicate it's FDR-corrected pairwise)
                    ax.text(pvalue_x, i, f"p={pval_corrected:.3e} (FDR)", ha='left', va='center',
                           fontsize=8, style='italic', color='darkblue')

        # Formatting with 2-level y-axis labels
        ax.set_xlabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(
            f"Mann-Whitney U Test: {metric}\n" +
            f"PRIMARY: Control vs ALL (1 test) | SECONDARY: Pairwise (FDR-corrected)",
            fontsize=13, fontweight="bold"
        )
        ax.set_yticks(range(len(hierarchical_groups)))
        ax.set_yticklabels(atr_labels, fontsize=9)  # Secondary labels (ATR condition)
        ax.grid(True, alpha=0.3, axis='x')

        # Add primary y-axis labels (Genotype) on the left side
        # Group positions by genotype
        genotype_positions = {}
        for i, genotype in enumerate(genotype_labels):
            if genotype not in genotype_positions:
                genotype_positions[genotype] = []
            genotype_positions[genotype].append(i)

        # Add genotype labels at the center of each genotype group
        ax2 = ax.secondary_yaxis('left')
        genotype_y_ticks = [np.mean(positions) for positions in genotype_positions.values()]
        genotype_names = list(genotype_positions.keys())
        ax2.set_yticks(genotype_y_ticks)
        ax2.set_yticklabels(genotype_names, fontsize=11, fontweight='bold')
        ax2.tick_params(length=0)  # Remove tick marks

        # Move primary y-label to secondary axis
        ax.set_ylabel("")
        ax2.set_ylabel("Genotype", fontsize=12, fontweight="bold", labelpad=10)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="lightgray", alpha=0.3, label="Control"),
            Patch(facecolor="lightgreen", alpha=0.3, label="Significantly increased"),
            Patch(facecolor="lightyellow", alpha=0.3, label="Not significant"),
            Patch(facecolor="lightcoral", alpha=0.3, label="Significantly decreased"),
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='ATR=yes'),
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='ATR=no'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd',
                   markersize=8, label='OR67d genotype'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f',
                   markersize=8, label='Other genotype'),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
                 fontsize=10, framealpha=0.9)

        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric}_combined_mannwhitney.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"{metric}_combined_mannwhitney.pdf", bbox_inches="tight")
        plt.close()

        metric_elapsed = time.time() - metric_start_time
        print(f"  ✅ Saved: {output_file} ({metric_elapsed:.2f}s)")

    # Create statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        stats_file = output_dir / "combined_mannwhitney_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\n✅ Statistics saved to: {stats_file}")

    return stats_df


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the Gtacr OR67d dataset"""

    dataset_path = (
        "/mnt/upramdya_data/MD/Gtacr/Datasets/251219_10_summary_OR67d_Gtacr_Data/summary/pooled_summary.feather"
    )

    print(f"Loading Gtacr OR67d dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise

    # Filter out experiments with Date == 230719
    dataset = dataset[dataset["Date"] != 230719]
    print(f"Dataset shape after filtering Date == 230719: {dataset.shape}")

    # Handle ATR column
    if "ATR" in dataset.columns:
        if pd.api.types.is_categorical_dtype(dataset["ATR"]):
            dataset["ATR"] = dataset["ATR"].astype(str)

        # Replace 'nan' string with actual NaN
        dataset.loc[dataset["ATR"] == "nan", "ATR"] = pd.NA

        # Fill NaN with "yes" (default ATR condition)
        dataset["ATR"] = dataset["ATR"].fillna("yes")

        print(f"\nATR values distribution:")
        print(dataset["ATR"].value_counts())
    else:
        print("⚠️  ATR column not found - creating with default 'yes'")
        dataset["ATR"] = "yes"

    # Harmonize genotype naming
    if "Genotype" in dataset.columns:
        genotype_mapping = {
            "OGS32xOR67d": "GtacrxOR67d",
            "OGS32xEmptyGal4": "GtacrxEmptyGal4",
        }

        original_genotypes = dataset["Genotype"].value_counts()
        print(f"\nGenotypes before harmonization:")
        print(original_genotypes)

        dataset["Genotype"] = dataset["Genotype"].replace(genotype_mapping)

        harmonized_genotypes = dataset["Genotype"].value_counts()
        print(f"\nGenotypes after harmonization:")
        print(harmonized_genotypes)

    # Drop metadata columns
    metadata_columns = [
        "Nickname", "Brain region", "Simplified Nickname", "Split",
        "Driver", "Experiment", "Date", "Arena", "BallType", "Dissected",
    ]

    columns_to_drop = [col for col in metadata_columns if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"\nDropped metadata columns: {columns_to_drop}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Test mode sampling
    if test_mode and len(dataset) > test_sample_size:
        original_shape = dataset.shape
        dataset = dataset.sample(n=test_sample_size, random_state=42)
        print(f"\n🧪 TEST MODE: Sampled {dataset.shape[0]} rows from {original_shape[0]}")

    return dataset


def should_skip_metric(metric, output_dir, overwrite):
    """Check if a metric should be skipped based on existing files"""
    if overwrite:
        return False

    output_file = output_dir / f"{metric}_combined_mannwhitney.png"
    return output_file.exists()


def main(overwrite=True, test_mode=False):
    """Main function to run combined ATR/no-ATR Mann-Whitney analysis"""

    print(f"{'='*80}")
    print("Starting Combined ATR/no-ATR Mann-Whitney Analysis for Gtacr OR67d")
    print(f"{'='*80}")

    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Show ATR x Genotype breakdown
    print("\n📊 Genotype x ATR distribution:")
    for genotype in sorted(dataset["Genotype"].unique()):
        print(f"  {genotype}:")
        atr_counts = dataset[dataset["Genotype"] == genotype]["ATR"].value_counts()
        for atr_val, count in atr_counts.items():
            print(f"    ATR={atr_val}: {count} samples")

    # Define output directory
    base_output_dir = Path("/mnt/upramdya_data/MD/Gtacr/Plots/summaries/Genotype_Combined_ATR")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {base_output_dir}")

    # Define metrics to analyze
    print(f"\n🎯 Defining metrics for analysis...")

    # Core base metrics
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
        "major_event",
        "major_event_time",
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_velocity",
        "auc",
        "overall_interaction_rate",
        "time_to_first_interaction",
    ]

    # Additional pattern-based metrics
    additional_patterns = [
        "velocity",
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
        "median_",
        "mean_",
    ]

    excluded_patterns = [
        "binned_",
        "r2",
        "slope",
        "_bin_",
        "logistic_",
    ]

    # Find metrics that exist in the dataset
    available_metrics = []

    # Check base metrics
    for metric in base_metrics:
        if metric in dataset.columns:
            available_metrics.append(metric)

    # Check pattern-based metrics
    for pattern in additional_patterns:
        for col in dataset.columns:
            if pattern in col.lower() and col not in available_metrics:
                # Check if it should be excluded
                if not any(excl in col.lower() for excl in excluded_patterns):
                    available_metrics.append(col)

    print(f"📊 Found {len(available_metrics)} metrics")

    # Categorize metrics
    def is_binary(series):
        unique_vals = series.dropna().unique()
        return len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False})

    continuous_metrics = []
    binary_metrics = []

    for col in available_metrics:
        if col in dataset.columns:
            if is_binary(dataset[col]):
                binary_metrics.append(col)
            else:
                continuous_metrics.append(col)

    print(f"\n📊 METRICS SUMMARY:")
    print(f"  Continuous metrics: {len(continuous_metrics)}")
    print(f"  Binary metrics: {len(binary_metrics)}")

    # Test mode filtering
    if test_mode:
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2]
        print(f"\n🧪 TEST MODE: Processing {len(continuous_metrics)} continuous + {len(binary_metrics)} binary metrics")

    # Process continuous metrics
    if continuous_metrics:
        print(f"\n{'='*80}")
        print(f"CONTINUOUS METRICS ANALYSIS ({len(continuous_metrics)} metrics)")
        print(f"{'='*80}")

        # Filter metrics based on overwrite setting
        metrics_to_process = []
        metrics_skipped = []

        for metric in continuous_metrics:
            if should_skip_metric(metric, base_output_dir, overwrite):
                metrics_skipped.append(metric)
            else:
                metrics_to_process.append(metric)

        if metrics_skipped:
            print(f"\n📄 Skipping {len(metrics_skipped)} metrics (already exist):")
            for metric in metrics_skipped[:5]:
                print(f"  - {metric}")
            if len(metrics_skipped) > 5:
                print(f"  ... and {len(metrics_skipped) - 5} more")

        if metrics_to_process:
            print(f"\n📊 Processing {len(metrics_to_process)} continuous metrics...")

            stats_df = generate_combined_mannwhitney_plots(
                data=dataset,
                metrics=metrics_to_process,
                genotype_col="Genotype",
                atr_col="ATR",
                control_genotype="GtacrxOR67d",
                control_atr="yes",
                figsize=(18, 10),
                output_dir=base_output_dir,
                fdr_method="fdr_bh",
                alpha=0.05,
            )

            print(f"\n✅ Continuous metrics analysis complete!")
        else:
            print(f"\n📄 All continuous metrics already processed (use --overwrite to regenerate)")

    # Process binary metrics if needed
    if binary_metrics:
        print(f"\n{'='*80}")
        print(f"NOTE: Binary metrics ({len(binary_metrics)} found) can be analyzed separately")
        print("Consider using Fisher's exact test for binary outcomes")
        print(f"{'='*80}")

    # Final summary
    print(f"\n{'='*80}")
    print("✅ COMBINED ATR/no-ATR ANALYSIS COMPLETE!")
    print(f"📁 Output directory: {base_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate combined ATR/no-ATR Mann-Whitney plots for Gtacr OR67d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_gtacr_combined.py                    # Overwrite existing plots
  python run_mannwhitney_gtacr_combined.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_gtacr_combined.py --test             # Test mode with 3 metrics
        """,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip metrics that already have existing plots (default: overwrite existing plots)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process first 3 metrics for debugging (default: process all metrics)",
    )

    args = parser.parse_args()

    # Run main function
    main(overwrite=not args.no_overwrite, test_mode=args.test)
