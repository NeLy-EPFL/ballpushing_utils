#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for max event metrics.
1. max_event by Pretraining condition
2. max_event_time by Pretraining condition
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def create_boxplot_with_scatter(data, x_col, y_col, title, ax, color_palette=None):
    box_plot = sns.boxplot(data=data, x=x_col, y=y_col, ax=ax, hue=x_col, palette=color_palette, legend=False)
    for patch in box_plot.patches:
        patch.set_alpha(0.7)
    sns.stripplot(data=data, x=x_col, y=y_col, ax=ax, size=4, alpha=0.6, jitter=True, dodge=False, color="black")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        condition = label.get_text()
        n = len(data[data[x_col] == condition])
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")
    unique_conditions = data[x_col].unique()
    if len(unique_conditions) >= 2:
        groups = [data[data[x_col] == condition][y_col] for condition in unique_conditions]
        if len(unique_conditions) == 2:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        else:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        ax.text(
            0.02,
            0.98,
            f"{test_name}: p = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
    return ax


def main():
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather"
    )
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")
    max_event_col = "max_event" if "max_event" in df.columns else None
    max_event_time_col = "max_event_time" if "max_event_time" in df.columns else None
    pretraining_col = None
    ball_condition_col = None
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break
    if "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"
    elif "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"
    else:
        for col in df.columns:
            if "ball" in col.lower() and ("condition" in col.lower() or "identity" in col.lower()):
                ball_condition_col = col
                break
    required_columns = {
        "max_event": max_event_col,
        "max_event_time": max_event_time_col,
        "pretraining": pretraining_col,
        "ball_condition": ball_condition_col,
    }
    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        return
    df_clean = df[[max_event_col, max_event_time_col, pretraining_col, ball_condition_col]]
    df_clean = df_clean.dropna(subset=[pretraining_col, ball_condition_col])
    test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
    test_ball_data = pd.DataFrame()
    for test_val in test_ball_values:
        test_subset = df_clean[df_clean[ball_condition_col] == test_val]
        if not test_subset.empty:
            test_ball_data = test_subset
            break
    if test_ball_data.empty:
        print("No specific 'test' ball data found. Using all data.")
        test_ball_data = df_clean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    pretraining_palette = sns.color_palette("Set2", n_colors=len(test_ball_data[pretraining_col].unique()))
    event_data = test_ball_data.dropna(subset=[max_event_col])
    if not event_data.empty:
        create_boxplot_with_scatter(
            data=event_data,
            x_col=pretraining_col,
            y_col=max_event_col,
            title="Max Event by Pretraining Condition",
            ax=ax1,
            color_palette=pretraining_palette,
        )
    else:
        ax1.text(0.5, 0.5, "No valid data for max_event", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Max Event by Pretraining Condition")
    time_data = test_ball_data.copy()
    mask = pd.isna(time_data[max_event_time_col]) | np.isinf(time_data[max_event_time_col])
    time_data_filtered = time_data[~mask]
    if len(time_data_filtered) >= 2:
        create_boxplot_with_scatter(
            data=time_data_filtered,
            x_col=pretraining_col,
            y_col=max_event_time_col,
            title="Max Event Time by Pretraining Condition",
            ax=ax2,
            color_palette=pretraining_palette,
        )
    else:
        ax2.text(
            0.5,
            0.5,
            f"Insufficient valid data\n({len(time_data_filtered)} valid points)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Max Event Time by Pretraining Condition")
        if pretraining_col:
            ax2.set_xlabel(pretraining_col.replace("_", " ").title())
        ax2.set_ylabel("Max Event Time")
    plt.tight_layout()
    output_path = Path(__file__).parent / "max_event_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    if pretraining_col and max_event_col:
        print(f"\n1. Max Event by {pretraining_col}:")
        event_data = test_ball_data.dropna(subset=[max_event_col])
        if not event_data.empty:
            event_summary = (
                event_data.groupby(pretraining_col)[max_event_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(event_summary)
        else:
            print("No valid data for max_event")
    if pretraining_col and max_event_time_col:
        print(f"\n2. Max Event Time by {pretraining_col}:")
        time_data = test_ball_data.dropna(subset=[max_event_time_col])
        if not time_data.empty:
            time_summary = (
                time_data.groupby(pretraining_col)[max_event_time_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(time_summary)
        else:
            print("No valid data for max_event_time")
    correlation_data = test_ball_data.dropna(subset=[max_event_col, max_event_time_col])
    if not correlation_data.empty and len(correlation_data) > 1:
        correlation, p_value = stats.pearsonr(correlation_data[max_event_col], correlation_data[max_event_time_col])
        print(f"\nCorrelation between max event and time (n={len(correlation_data)}):")
        print(f"Pearson r = {correlation:.3f}, p = {p_value:.4f}")
    else:
        print(f"\nInsufficient valid data for correlation analysis (n={len(correlation_data)})")
    plt.show()
    return test_ball_data


if __name__ == "__main__":
    data = main()
