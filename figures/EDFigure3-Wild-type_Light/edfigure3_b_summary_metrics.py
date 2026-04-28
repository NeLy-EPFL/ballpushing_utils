#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 3b: Summary metrics for light conditions.

This script generates boxplot comparisons showing:
- First major event time (displacement > 1.2 mm)
- Time spent in starting chamber
- Area under the displacement curve (AUC)
- Comparison between Light ON (control) and Light OFF
- Permutation tests with significance annotations

Usage:
    python edfigure3_b_summary_metrics.py [--test]
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

# Add src directory to path for imports

from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# Resolve the path at module level — putting the ``dataset(...)`` call
# inside ``load_and_clean_exploration_dataset()`` would conflict with
# the local ``dataset = read_feather(...)`` assignment on line 101
# (Python sees one assignment to ``dataset`` anywhere in the function
# and treats the name as local for the whole function, raising
# UnboundLocalError on the earlier ``dataset(...)`` call).
SUMMARY_PATH = dataset(
    "Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/summary/pooled_summary.feather"
)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform two-tailed permutation test on median difference.
    This matches the approach in run_permutation_light_conditions.py

    Parameters:
    -----------
    group1 : array-like
        Control group values
    group2 : array-like
        Test group values
    n_permutations : int
        Number of permutations

    Returns:
    --------
    float
        Two-tailed p-value
    """
    if len(group1) == 0 or len(group2) == 0:
        return 1.0

    # Observed statistic: median difference (robust to outliers)
    obs_stat = float(np.median(group2) - np.median(group1))

    # Permutation test
    combined = np.concatenate([group1, group2])
    n_control = len(group1)

    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_control = combined[:n_control]
        perm_test = combined[n_control:]
        perm_diffs[i] = np.median(perm_test) - np.median(perm_control)

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))

    return p_value


def load_and_clean_exploration_dataset(test_mode=False):
    """Load and clean the exploration dataset with light conditions

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    """
    # Load the exploration dataset
    print(f"Loading exploration dataset from: {SUMMARY_PATH}")
    try:
        dataset = read_feather(SUMMARY_PATH)
        print(f"Loaded dataset with shape: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {SUMMARY_PATH}")

    # Drop TNT-specific metadata columns
    tnt_metadata_columns = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Genotype",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
        "Period",
        "Condition",
        "Orientation",
        "Crossing",
        "BallType",
        "Used_to",
        "Magnet",
        "Peak",
    ]
    columns_to_drop = [col for col in tnt_metadata_columns if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"Dropped {len(columns_to_drop)} TNT metadata columns")

    # Clean up Light column
    print(f"Original Light column values: {sorted(dataset['Light'].unique())}")
    dataset = dataset[dataset["Light"] != ""].copy()
    print(f"After removing empty Light values, shape: {dataset.shape}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time", "index"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)

    print(f"Dataset cleaning completed successfully")

    # Add test mode sampling
    if test_mode and len(dataset) > 200:
        print(f"TEST MODE: Sampling 200 rows from {len(dataset)} total rows")
        dataset = dataset.sample(n=200, random_state=42)

    return dataset


def create_boxplot_comparison(
    data,
    metrics,
    metric_labels,
    control_condition="on",
    test_condition="off",
    group_col="Light",
    output_path=None,
):
    """
    Create a multi-panel boxplot comparison for selected metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing metrics
    metrics : list
        List of metric column names
    metric_labels : dict
        Dictionary mapping metric names to display labels
    control_condition : str
        Name of control condition
    test_condition : str
        Name of test condition
    group_col : str
        Column name for grouping
    output_path : Path or str
        Where to save the plot
    """
    # Filter data to only include the two conditions
    plot_data = data[data[group_col].isin([control_condition, test_condition])].copy()

    # Create figure with subplots
    n_metrics = len(metrics)

    if n_metrics == 1:
        # Single metric - simpler layout
        fig, ax = plt.subplots(figsize=(5, 6))
        axes = [ax]
    else:
        # Multiple metrics - multi-panel layout
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

    # Font sizes
    font_size_ticks = 10
    font_size_labels = 12
    font_size_title = 14

    # Colors: Light ON = orange, Light OFF = gray
    colors = {"on": "orange", "off": "gray"}

    # Store statistics
    stats_results = []

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get data for each condition
        control_data = plot_data[plot_data[group_col] == control_condition][metric].dropna()
        test_data = plot_data[plot_data[group_col] == test_condition][metric].dropna()

        # Prepare data for boxplot
        box_data = [test_data, control_data]
        positions = [1, 2]
        box_colors = [colors[test_condition], colors[control_condition]]

        # Create boxplot
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add jittered points
        np.random.seed(42)
        for i, (pos, box_values) in enumerate(zip(positions, box_data)):
            jitter = np.random.normal(0, 0.04, size=len(box_values))
            ax.scatter(
                pos + jitter,
                box_values,
                alpha=0.4,
                s=20,
                color=box_colors[i],
                edgecolors="none",
            )

        # Perform permutation test
        p_value = permutation_test(control_data, test_data, n_permutations=10000)

        # Calculate effect size
        effect_size = cohens_d(control_data, test_data)

        # Store statistics
        stats_results.append(
            {
                "metric": metric,
                "n_light_off": len(test_data),
                "n_light_on": len(control_data),
                "median_light_off": np.median(test_data),
                "median_light_on": np.median(control_data),
                "mean_light_off": np.mean(test_data),
                "mean_light_on": np.mean(control_data),
                "p_value": p_value,
                "cohens_d": effect_size,
            }
        )

        # Add significance annotation
        y_max = max(control_data.max(), test_data.max())
        y_range = y_max - min(control_data.min(), test_data.min())
        y_annot = y_max + 0.1 * y_range

        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"

        # Draw significance line
        ax.plot([1, 2], [y_annot, y_annot], "k-", linewidth=1.5)
        ax.text(1.5, y_annot, sig_marker, ha="center", va="bottom", fontsize=12)

        # Format axes
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                f"Light {test_condition.upper()}\n(n={len(test_data)})",
                f"Light {control_condition.upper()}\n(n={len(control_data)})",
            ],
            fontsize=font_size_ticks,
        )
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=font_size_labels)
        ax.tick_params(axis="y", labelsize=font_size_ticks)

        # Set y-axis limits to accommodate annotation
        ax.set_ylim(min(control_data.min(), test_data.min()) - 0.05 * y_range, y_annot + 0.15 * y_range)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save plot
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved boxplot to: {output_path}")

    plt.close()

    return pd.DataFrame(stats_results)


def main():
    """Main function for Extended Data Figure 3b"""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 3b: Light condition summary metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")

    args = parser.parse_args()

    # Output directory
    output_dir = figure_output_dir("EDFigure3", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 3b: Light Condition Summary Metrics")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading dataset...")
    data = load_and_clean_exploration_dataset(test_mode=args.test)

    # Filter to only FeedingState starved_noWater (CRITICAL for matching published results)
    if "FeedingState" in data.columns:
        original_len = len(data)
        data = data[data["FeedingState"] == "starved_noWater"].copy()
        print(f"Filtered for starved_noWater: {len(data)} flies (was {original_len})")

    # Filter to only Light on/off conditions
    data = data[data["Light"].isin(["on", "off"])].copy()
    print(f"After filtering to on/off conditions: {data.shape}")
    print(f"Light ON: {len(data[data['Light'] == 'on'])} flies")
    print(f"Light OFF: {len(data[data['Light'] == 'off'])} flies")

    # Convert time metrics from seconds to minutes
    time_metrics = ["first_major_event_time", "chamber_time", "chamber_exit_time"]

    print("\nConverting time metrics from seconds to minutes...")
    for metric in time_metrics:
        if metric in data.columns:
            data[metric] = data[metric] / 60.0
            print(f"  Converted {metric}")

    # Panel (b) shows ONLY first_major_event_time
    metric_to_plot = "first_major_event_time"

    if metric_to_plot not in data.columns:
        raise ValueError(f"Required metric '{metric_to_plot}' not found in dataset")

    # Define metric labels
    metric_labels = {
        "first_major_event_time": "First major event time (min)",
    }

    # Generate boxplot and compute statistics for first_major_event_time
    print(f"\nGenerating boxplot for first_major_event_time...")
    plot_file = output_dir / "edfigure3b_first_major_event_time.pdf"

    stats_df = create_boxplot_comparison(
        data,
        metrics=[metric_to_plot],
        metric_labels=metric_labels,
        control_condition="on",
        test_condition="off",
        group_col="Light",
        output_path=plot_file,
    )

    # Save statistics
    stats_file = output_dir / "edfigure3b_summary_statistics.csv"
    stats_df.to_csv(stats_file, index=False, float_format="%.6f")
    print(f"✅ Statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Statistical Summary:")
    print(f"{'='*60}")

    row = stats_df.iloc[0]
    print(f"\n{metric_labels.get(row['metric'], row['metric'])}:")
    print(
        f"  Light OFF: median={row['median_light_off']:.3f}, mean={row['mean_light_off']:.3f}, n={int(row['n_light_off'])}"
    )
    print(
        f"  Light ON:  median={row['median_light_on']:.3f}, mean={row['mean_light_on']:.3f}, n={int(row['n_light_on'])}"
    )
    print(f"  Permutation test p-value: {row['p_value']:.6f}")
    print(f"  Cohen's d: {row['cohens_d']:.3f}")

    if row["p_value"] < 0.001:
        sig = "***"
    elif row["p_value"] < 0.01:
        sig = "**"
    elif row["p_value"] < 0.05:
        sig = "*"
    else:
        sig = "ns"
    print(f"  Significance: {sig}")

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 3b generated successfully!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  - Plot: {plot_file}")
    print(f"  - Statistics: {stats_file}")
    print(f"\nNote: Only first_major_event_time is plotted and reported.")
    print(f"      Expected p-value with full dataset: ~0.039")
    print(f"      Permutation test uses median difference (matching original analysis)")


if __name__ == "__main__":
    main()
