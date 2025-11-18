#!/usr/bin/env python3
"""
Script to create a scatter plot of training vs test ball distance moved for Pretraining == "y" flies.
X-axis: distance_moved for ball_condition "training"
Y-axis: distance_moved for ball_condition "test"
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    """Main function to create the training vs test scatter plot."""

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

    # Remove the data for Date 250904

    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")
    else:
        print("Column 'Date' not found in dataset. Skipping date-based filtering.")

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # --- Generalize: Find all metrics that exist for both test and training ball conditions ---
    # Identify key columns

    # Use explicit column names based on your dataset
    pretraining_col = "Pretraining"
    ball_condition_col = "ball_identity"
    f1_condition_col = "F1_condition"

    if not all([pretraining_col, ball_condition_col, f1_condition_col]):
        print("Could not find required columns for pretraining, ball_condition, or f1_condition.")
        return

    # Add fly identifier column if it doesn't exist
    fly_id_col = "fly"
    if fly_id_col not in df.columns:
        fly_cols = [col for col in df.columns if "fly" in col.lower()]
        if fly_cols:
            fly_id_col = fly_cols[0]
            print(f"Using '{fly_id_col}' as fly identifier column")
        else:
            print("Warning: No fly identifier column found. Using row index.")
            df["fly"] = df.index
            fly_id_col = "fly"

    # Find all metric columns that are numeric and not grouping columns
    exclude_cols = {pretraining_col, ball_condition_col, f1_condition_col, fly_id_col, "Date"}
    metric_candidates = [
        col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # For each metric, check if it exists for both test and training ball conditions
    # We'll do this by pivoting for each metric and checking columns
    available_metrics = []
    for metric in metric_candidates:
        pivot = df.pivot_table(index=fly_id_col, columns=ball_condition_col, values=metric, aggfunc="first")
        if "training" in pivot.columns and "test" in pivot.columns:
            available_metrics.append(metric)

    if not available_metrics:
        print("No metrics found that exist for both test and training ball conditions.")
        return

    print(f"Metrics available for scatter: {available_metrics}")

    # --- For each metric, plot by F1_condition for all unique pretraining groups in the data ---
    colors = ["steelblue", "orange", "green", "red", "purple"]

    # Get all unique pretraining group values in the data
    pretraining_groups = df[pretraining_col].dropna().unique()

    for metric in available_metrics:
        print(f"\n==== Metric: {metric} ====")
        for pretrain_val in pretraining_groups:
            group_df = df[df[pretraining_col] == pretrain_val]
            if group_df.empty:
                print(f"No flies with {pretraining_col} == '{pretrain_val}' for metric {metric}.")
                continue
            f1_conditions = group_df[f1_condition_col].unique()
            n_conditions = len(f1_conditions)
            fig, axes = plt.subplots(1, n_conditions + 1, figsize=(6 * (n_conditions + 1), 6))
            if n_conditions == 0:
                axes = [axes]
            elif n_conditions == 1:
                axes = [axes[0], axes[1]]
            results = {}
            all_complete_data = []
            for i, f1_condition in enumerate(f1_conditions):
                ax = axes[i]
                condition_data = group_df[group_df[f1_condition_col] == f1_condition]
                pivot_data = condition_data.pivot_table(
                    index=fly_id_col,
                    columns=ball_condition_col,
                    values=metric,
                    aggfunc="first",
                )
                if "training" not in pivot_data.columns or "test" not in pivot_data.columns:
                    ax.text(0.5, 0.5, f"No data for {f1_condition}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"F1_condition: {f1_condition}")
                    continue
                complete_data = pivot_data.dropna(subset=["training", "test"])
                if complete_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No complete data for {f1_condition}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"F1_condition: {f1_condition}")
                    continue
                complete_data_with_condition = complete_data.copy()
                complete_data_with_condition["F1_condition"] = f1_condition
                all_complete_data.append(complete_data_with_condition)
                color = colors[i % len(colors)]
                x = complete_data["training"]
                y = complete_data["test"]
                # Convert boolean columns to int for plotting and calculations
                if pd.api.types.is_bool_dtype(x):
                    x = x.astype(int)
                if pd.api.types.is_bool_dtype(y):
                    y = y.astype(int)
                ax.scatter(x, y, alpha=0.7, s=60, edgecolors="black", linewidth=0.5, color=color)
                if len(x) > 1 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                    try:
                        correlation_coef, p_value = stats.pearsonr(x, y)
                        x_normalized = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
                        y_normalized = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
                        normalized_correlation_coef = float("nan")
                        normalized_p_value = float("nan")
                        if x_normalized.std() > 0 and y_normalized.std() > 0:
                            normalized_correlation_coef, normalized_p_value = stats.pearsonr(x_normalized, y_normalized)
                        z = np.polyfit(x, y, 1)
                        fit_line = np.poly1d(z)
                        ax.plot(
                            x,
                            fit_line(x),
                            color=color,
                            alpha=0.8,
                            linewidth=2,
                            label=f"Linear fit (r = {correlation_coef:.3f})",
                        )
                        results[f1_condition] = {
                            "correlation": correlation_coef,
                            "p_value": p_value,
                            "normalized_correlation": normalized_correlation_coef,
                            "normalized_p_value": normalized_p_value,
                            "slope": z[0],
                            "intercept": z[1],
                            "n_flies": len(complete_data),
                            "x_data": x,
                            "y_data": y,
                        }
                        textstr = (
                            f"Raw: r = {correlation_coef:.3f} (p = {p_value:.3f})\n"
                            f"Norm: r = {normalized_correlation_coef:.3f} (p = {normalized_p_value:.3f})\n"
                            f"n = {len(complete_data)}"
                        )
                        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
                        ax.text(
                            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=props
                        )
                    except Exception as e:
                        ax.text(
                            0.05,
                            0.95,
                            f"No fit: {e}",
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                        )
                        results[f1_condition] = {
                            "correlation": np.nan,
                            "p_value": np.nan,
                            "normalized_correlation": np.nan,
                            "normalized_p_value": np.nan,
                            "slope": np.nan,
                            "intercept": np.nan,
                            "n_flies": len(complete_data),
                            "x_data": x,
                            "y_data": y,
                        }
                else:
                    ax.text(
                        0.05,
                        0.95,
                        "No fit: constant or insufficient data",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )
                    results[f1_condition] = {
                        "correlation": np.nan,
                        "p_value": np.nan,
                        "normalized_correlation": np.nan,
                        "normalized_p_value": np.nan,
                        "slope": np.nan,
                        "intercept": np.nan,
                        "n_flies": len(complete_data),
                        "x_data": x,
                        "y_data": y,
                    }
                ax.set_xlabel(f"{metric} - Training Ball", fontsize=11)
                ax.set_ylabel(f"{metric} - Test Ball", fontsize=11)
                ax.set_title(f"F1_condition: {f1_condition}", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                if len(x) > 0 and len(y) > 0:
                    x_range = x.max() - x.min() if x.max() > x.min() else x.mean() * 0.1
                    y_range = y.max() - y.min() if y.max() > y.min() else y.mean() * 0.1
                    x_padding = x_range * 0.1
                    y_padding = y_range * 0.1
                    ax.set_xlim(x.min() - x_padding, x.max() + x_padding)
                    ax.set_ylim(y.min() - y_padding, y.max() + y_padding)
            # Save and print output path for each plot
            out_path = f"scatter_{metric}_pretraining_{pretrain_val}.png"
            fig.tight_layout()
            fig.savefig(out_path)
            print(f"Plot saved to: {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    data = main()
