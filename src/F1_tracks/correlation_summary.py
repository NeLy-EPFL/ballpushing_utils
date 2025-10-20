#!/usr/bin/env python3
"""
Summary script to compare normalized correlations across all behavioral metrics.
This script provides a consolidated view of how well training performance predicts
test performance for different metrics, accounting for experimental constraints.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def analyze_metric_correlations(df, metric_name, metric_col):
    """Analyze correlations for a specific metric."""

    # Required columns
    required_cols = [metric_col, "Pretraining", "ball_identity", "F1_condition", "fly"]

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for {metric_name}: {missing_cols}")
        return None

    # Clean the data
    df_clean = df[required_cols].dropna()

    # Filter for Pretraining == "y" flies
    pretrained_flies = df_clean[df_clean["Pretraining"] == "y"]

    if pretrained_flies.empty:
        print(f"No pretrained flies found for {metric_name}")
        return None

    # Get F1_conditions
    f1_conditions = ["pretrained", "pretrained_unlocked"]
    available_conditions = [cond for cond in f1_conditions if cond in pretrained_flies["F1_condition"].unique()]

    results = {}

    for condition in available_conditions:
        # Filter data for this F1_condition
        condition_data = pretrained_flies[pretrained_flies["F1_condition"] == condition]

        # Pivot the data to get training and test values for each fly
        pivot_data = condition_data.pivot_table(
            index="fly", columns="ball_identity", values=metric_col, aggfunc="first"
        )

        # Check if we have both training and test conditions
        if "training" not in pivot_data.columns or "test" not in pivot_data.columns:
            continue

        # Remove flies that don't have both training and test data
        complete_data = pivot_data.dropna(subset=["training", "test"])

        if complete_data.empty or len(complete_data) < 2:
            continue

        x = complete_data["training"]
        y = complete_data["test"]

        # Calculate correlations
        raw_corr, raw_p = stats.pearsonr(x, y)

        # Normalized correlation
        x_normalized = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
        y_normalized = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0

        if x_normalized.std() > 0 and y_normalized.std() > 0:
            norm_corr, norm_p = stats.pearsonr(x_normalized, y_normalized)
        else:
            norm_corr, norm_p = float("nan"), float("nan")

        results[condition] = {
            "metric": metric_name,
            "n_flies": len(complete_data),
            "raw_correlation": raw_corr,
            "raw_p_value": raw_p,
            "normalized_correlation": norm_corr,
            "normalized_p_value": norm_p,
            "training_mean": x.mean(),
            "training_std": x.std(),
            "training_range": (x.min(), x.max()),
            "test_mean": y.mean(),
            "test_std": y.std(),
            "test_range": (y.min(), y.max()),
            "mean_ratio": y.mean() / x.mean() if x.mean() != 0 else float("nan"),
        }

    return results


def main():
    """Main function to create correlation summary."""

    # Dataset path
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Load dataset
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Remove Date 250904 if present
    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")

    # Define metrics to analyze
    metrics = [
        ("Distance Moved", "distance_moved"),
        ("Max Distance", "max_distance"),
        ("Pulling Ratio", "pulling_ratio"),
        ("Pulled Events", "pulled"),
    ]

    # Analyze each metric
    all_results = {}

    for metric_name, metric_col in metrics:
        print(f"\n{'='*50}")
        print(f"Analyzing {metric_name}")
        print(f"{'='*50}")

        results = analyze_metric_correlations(df, metric_name, metric_col)
        if results:
            all_results[metric_name] = results

            for condition, data in results.items():
                print(f"\n{condition}:")
                print(f"  n = {data['n_flies']} flies")
                print(
                    f"  Training: {data['training_mean']:.3f} ± {data['training_std']:.3f} (range: {data['training_range'][0]:.1f}-{data['training_range'][1]:.1f})"
                )
                print(
                    f"  Test: {data['test_mean']:.3f} ± {data['test_std']:.3f} (range: {data['test_range'][0]:.1f}-{data['test_range'][1]:.1f})"
                )
                print(f"  Raw correlation: r = {data['raw_correlation']:.3f} (p = {data['raw_p_value']:.3f})")
                print(
                    f"  Normalized correlation: r = {data['normalized_correlation']:.3f} (p = {data['normalized_p_value']:.3f})"
                )
                print(f"  Test/Training ratio: {data['mean_ratio']:.3f}")

    # Create summary visualization
    if all_results:
        create_correlation_summary_plot(all_results)

    return all_results


def create_correlation_summary_plot(all_results):
    """Create a summary plot comparing correlations across metrics."""

    # Prepare data for plotting
    plot_data = []

    for metric_name, conditions in all_results.items():
        for condition, data in conditions.items():
            plot_data.append(
                {
                    "Metric": metric_name,
                    "F1_condition": condition,
                    "Raw_Correlation": data["raw_correlation"],
                    "Normalized_Correlation": data["normalized_correlation"],
                    "Raw_P_Value": data["raw_p_value"],
                    "Normalized_P_Value": data["normalized_p_value"],
                    "N_Flies": data["n_flies"],
                    "Test_Training_Ratio": data["mean_ratio"],
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Raw vs Normalized Correlations
    ax1 = axes[0, 0]
    for condition in ["pretrained", "pretrained_unlocked"]:
        condition_data = plot_df[plot_df["F1_condition"] == condition]
        ax1.scatter(
            condition_data["Raw_Correlation"],
            condition_data["Normalized_Correlation"],
            label=condition.replace("_", " ").title(),
            alpha=0.7,
            s=100,
        )

        # Add metric labels
        for _, row in condition_data.iterrows():
            ax1.annotate(
                row["Metric"][:8],
                (row["Raw_Correlation"], row["Normalized_Correlation"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # Add diagonal line
    min_val = min(plot_df["Raw_Correlation"].min(), plot_df["Normalized_Correlation"].min())
    max_val = max(plot_df["Raw_Correlation"].max(), plot_df["Normalized_Correlation"].max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Equal correlation")

    ax1.set_xlabel("Raw Correlation")
    ax1.set_ylabel("Normalized Correlation")
    ax1.set_title("Raw vs Normalized Correlations")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correlation Comparison by Metric
    ax2 = axes[0, 1]
    metrics = plot_df["Metric"].unique()
    x_pos = np.arange(len(metrics))
    width = 0.35

    pretrained_raw = []
    pretrained_norm = []
    unlocked_raw = []
    unlocked_norm = []

    for metric in metrics:
        metric_data = plot_df[plot_df["Metric"] == metric]

        pretrained_data = metric_data[metric_data["F1_condition"] == "pretrained"]
        unlocked_data = metric_data[metric_data["F1_condition"] == "pretrained_unlocked"]

        pretrained_raw.append(pretrained_data["Raw_Correlation"].iloc[0] if len(pretrained_data) > 0 else 0)
        pretrained_norm.append(pretrained_data["Normalized_Correlation"].iloc[0] if len(pretrained_data) > 0 else 0)
        unlocked_raw.append(unlocked_data["Raw_Correlation"].iloc[0] if len(unlocked_data) > 0 else 0)
        unlocked_norm.append(unlocked_data["Normalized_Correlation"].iloc[0] if len(unlocked_data) > 0 else 0)

    ax2.bar(x_pos - width / 2, pretrained_raw, width / 2, label="Pretrained Raw", alpha=0.7)
    ax2.bar(
        x_pos - width / 2, pretrained_norm, width / 2, bottom=pretrained_raw, label="Pretrained Norm (diff)", alpha=0.7
    )
    ax2.bar(x_pos + width / 2, unlocked_raw, width / 2, label="Unlocked Raw", alpha=0.7)
    ax2.bar(x_pos + width / 2, unlocked_norm, width / 2, bottom=unlocked_raw, label="Unlocked Norm (diff)", alpha=0.7)

    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Correlation Coefficient")
    ax2.set_title("Correlation by Metric and Condition")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m[:8] for m in metrics], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Test/Training Ratio by Metric
    ax3 = axes[1, 0]
    for condition in ["pretrained", "pretrained_unlocked"]:
        condition_data = plot_df[plot_df["F1_condition"] == condition]
        ax3.bar(
            condition_data["Metric"],
            condition_data["Test_Training_Ratio"],
            alpha=0.7,
            label=condition.replace("_", " ").title(),
        )

    ax3.set_xlabel("Metric")
    ax3.set_ylabel("Test/Training Ratio")
    ax3.set_title("Test vs Training Performance Ratio")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sample Size by Metric
    ax4 = axes[1, 1]
    for condition in ["pretrained", "pretrained_unlocked"]:
        condition_data = plot_df[plot_df["F1_condition"] == condition]
        ax4.bar(
            condition_data["Metric"], condition_data["N_Flies"], alpha=0.7, label=condition.replace("_", " ").title()
        )

    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Number of Flies")
    ax4.set_title("Sample Size by Metric")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "correlation_summary_all_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSummary plot saved to: {output_path}")

    # Create summary table
    print(f"\n{'='*80}")
    print("CORRELATION SUMMARY TABLE")
    print(f"{'='*80}")
    print(
        f"{'Metric':<15} {'Condition':<20} {'n':<3} {'Raw r':<8} {'Norm r':<8} {'Raw p':<8} {'Norm p':<8} {'Ratio':<6}"
    )
    print("-" * 80)

    for metric_name, conditions in all_results.items():
        for condition, data in conditions.items():
            print(
                f"{metric_name:<15} {condition:<20} {data['n_flies']:<3} "
                f"{data['raw_correlation']:<8.3f} {data['normalized_correlation']:<8.3f} "
                f"{data['raw_p_value']:<8.3f} {data['normalized_p_value']:<8.3f} "
                f"{data['mean_ratio']:<6.3f}"
            )


if __name__ == "__main__":
    results = main()
