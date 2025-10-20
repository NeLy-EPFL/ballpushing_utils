#!/usr/bin/env python3
"""
Script to create a scatter plot of training vs test ball pulled events for Pretraining == "y" flies.
X-axis: pulled for ball_condition "training"
Y-axis: pulled for ball_condition "test"
Compare across different F1_condition values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    """Main function to create the training vs test pulled events scatter plot."""

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

    # Try to identify the correct column names
    pulled_col = None
    pretraining_col = None
    ball_condition_col = None
    f1_condition_col = None

    # Look for pulled column
    for col in df.columns:
        if col.lower() == "pulled":  # Exact match
            pulled_col = col
            break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower() or "training" in col.lower():
            pretraining_col = col
            break

    # Look for ball condition column - prioritize ball_condition over ball_identity
    if "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"
    elif "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"
    else:
        for col in df.columns:
            if "ball" in col.lower() and ("condition" in col.lower() or "identity" in col.lower()):
                ball_condition_col = col
                break

    # Look for F1_condition column
    for col in df.columns:
        if "f1" in col.lower() and "condition" in col.lower():
            f1_condition_col = col
            break

    # If exact matches not found, use manual mapping based on your dataset structure
    if pulled_col is None:
        # Look for any pull-related column
        pull_cols = [col for col in df.columns if "pull" in col.lower()]
        if pull_cols:
            pulled_col = pull_cols[0]
            print(f"Using '{pulled_col}' as pulled events column")

    if pretraining_col is None:
        # Look for any training-related column
        train_cols = [col for col in df.columns if "train" in col.lower()]
        if train_cols:
            pretraining_col = train_cols[0]
            print(f"Using '{pretraining_col}' as pretraining column")

    if ball_condition_col is None:
        # Check if there's a direct 'ball_condition' column
        if "ball_condition" in df.columns:
            ball_condition_col = "ball_condition"
        else:
            # Look for any ball-related column
            ball_cols = [col for col in df.columns if "ball" in col.lower()]
            if ball_cols:
                ball_condition_col = ball_cols[0]
                print(f"Using '{ball_condition_col}' as ball condition column")

    # Check if we found all required columns
    required_columns = {
        "pulled": pulled_col,
        "pretraining": pretraining_col,
        "ball_condition": ball_condition_col,
        "f1_condition": f1_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  Pulled events: {pulled_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Ball condition: {ball_condition_col}")
    print(f"  F1 condition: {f1_condition_col}")

    # Add fly identifier column if it doesn't exist
    fly_id_col = "fly"
    if fly_id_col not in df.columns:
        # Look for fly identifier column
        fly_cols = [col for col in df.columns if "fly" in col.lower()]
        if fly_cols:
            fly_id_col = fly_cols[0]
            print(f"Using '{fly_id_col}' as fly identifier column")
        else:
            print("Warning: No fly identifier column found. Using row index.")
            df["fly"] = df.index
            fly_id_col = "fly"

    # Clean the data
    df_clean = df[[pulled_col, pretraining_col, ball_condition_col, f1_condition_col, fly_id_col]].dropna()
    print(f"Data shape after removing NaNs: {df_clean.shape}")

    # Debug: Show some sample data to verify the pulled values
    sample_data = df_clean[[pulled_col, ball_condition_col, f1_condition_col, fly_id_col]].head(10)
    print(f"\nSample data to verify pulled values:")
    print(sample_data)

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_clean[pretraining_col].unique()}")
    print(f"Unique {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")
    print(f"Unique {f1_condition_col} values: {df_clean[f1_condition_col].unique()}")

    # Print basic statistics for pulled events
    print(f"\nPulled events statistics:")
    print(f"  Overall range: {df_clean[pulled_col].min()} - {df_clean[pulled_col].max()}")
    print(f"  Mean: {df_clean[pulled_col].mean():.2f}")
    print(f"  Median: {df_clean[pulled_col].median():.2f}")

    # Filter for Pretraining == "y" flies
    pretrained_flies = df_clean[df_clean[pretraining_col] == "y"]
    print(f"Number of pretrained flies: {len(pretrained_flies[fly_id_col].unique())}")

    if pretrained_flies.empty:
        print("No flies with Pretraining == 'y' found. Available pretraining values:")
        print(df_clean[pretraining_col].value_counts())
        return

    # Print F1_condition distribution for pretrained flies
    print(f"F1_condition distribution for pretrained flies:")
    print(pretrained_flies[f1_condition_col].value_counts())

    # Get unique F1 conditions for pretrained flies
    f1_conditions = pretrained_flies[f1_condition_col].unique()
    print(f"F1 conditions found: {f1_conditions}")

    # Create a figure with subplots for each F1_condition
    n_conditions = len(f1_conditions)
    fig, axes = plt.subplots(1, n_conditions + 1, figsize=(6 * (n_conditions + 1), 6))

    # Ensure axes is always a list for consistent indexing
    if n_conditions == 0:
        axes = [axes]
    elif n_conditions == 1:
        axes = [axes[0], axes[1]]

    # Store results for comparison
    results = {}
    all_complete_data = []

    # Define colors for consistency across plots
    colors = ["steelblue", "orange", "green", "red", "purple"]

    # Plot for each F1_condition separately
    for i, f1_condition in enumerate(f1_conditions):
        ax = axes[i]

        # Filter data for this F1_condition
        condition_data = pretrained_flies[pretrained_flies[f1_condition_col] == f1_condition]

        # Pivot the data to get training and test pulled events for each fly
        pivot_data = condition_data.pivot_table(
            index=fly_id_col,
            columns=ball_condition_col,
            values=pulled_col,
            aggfunc="first",  # In case there are duplicates, take the first value
        )

        print(f"\nF1_condition '{f1_condition}':")
        print(f"  Pivot table shape: {pivot_data.shape}")
        print(f"  Pivot table columns: {pivot_data.columns.tolist()}")

        # Check if we have both training and test conditions
        if "training" not in pivot_data.columns or "test" not in pivot_data.columns:
            print(f"  Missing 'training' or 'test' ball conditions for {f1_condition}")
            ax.text(0.5, 0.5, f"No data for {f1_condition}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"F1_condition: {f1_condition}")
            continue

        # Remove flies that don't have both training and test data
        complete_data = pivot_data.dropna(subset=["training", "test"])
        print(f"  Number of flies with both training and test data: {len(complete_data)}")

        if complete_data.empty:
            print(f"  No flies have both training and test data for {f1_condition}")
            ax.text(0.5, 0.5, f"No complete data for {f1_condition}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"F1_condition: {f1_condition}")
            continue

        # Store complete data with F1_condition label
        complete_data_with_condition = complete_data.copy()
        complete_data_with_condition["f1_condition"] = f1_condition
        all_complete_data.append(complete_data_with_condition)

        # Set color for this condition
        color = colors[i % len(colors)]

        # Create scatter plot for this condition
        x = complete_data["training"]
        y = complete_data["test"]

        ax.scatter(x, y, alpha=0.7, s=60, edgecolors="black", linewidth=0.5, color=color)

        # Calculate correlation and trend line
        if len(x) > 1:  # Need at least 2 points for correlation
            # Check if there's variation in the data
            if x.std() > 0 and y.std() > 0:
                correlation_coef, p_value = stats.pearsonr(x, y)  # type: ignore

                # Add trend line (best fit)
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

                # Store results
                results[f1_condition] = {
                    "correlation": correlation_coef,
                    "p_value": p_value,
                    "slope": z[0],
                    "intercept": z[1],
                    "n_flies": len(complete_data),
                    "x_data": x,
                    "y_data": y,
                }

                # Add correlation text box
                r_squared = correlation_coef * correlation_coef  # type: ignore
                textstr = (
                    f"r = {correlation_coef:.3f}\n"
                    f"R² = {r_squared:.3f}\n"
                    f"p = {p_value:.3f}\n"
                    f"n = {len(complete_data)}"
                )
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)
            else:
                # No variation in data
                print(f"  Warning: No variation in pulled events for {f1_condition}")
                results[f1_condition] = {
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "n_flies": len(complete_data),
                    "x_data": x,
                    "y_data": y,
                }

                # Add text indicating no variation
                ax.text(
                    0.05,
                    0.95,
                    f"No variation\nn = {len(complete_data)}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                )
        else:
            results[f1_condition] = {
                "correlation": np.nan,
                "p_value": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "n_flies": len(complete_data),
                "x_data": x,
                "y_data": y,
            }

        # Formatting for individual plots
        ax.set_xlabel("Pulled Events - Training Ball", fontsize=11)
        ax.set_ylabel("Pulled Events - Test Ball", fontsize=11)
        ax.set_title(f"F1_condition: {f1_condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Set axis limits with padding (pulled events are integers >= 0)
        if len(x) > 0 and len(y) > 0:
            x_range = x.max() - x.min() if x.max() > x.min() else 1
            y_range = y.max() - y.min() if y.max() > y.min() else 1
            x_padding = max(x_range * 0.1, 0.5)  # At least 0.5 padding for integer data
            y_padding = max(y_range * 0.1, 0.5)  # At least 0.5 padding for integer data

            ax.set_xlim(max(0, x.min() - x_padding), x.max() + x_padding)
            ax.set_ylim(max(0, y.min() - y_padding), y.max() + y_padding)

        # Set integer ticks for better readability
        ax.set_xticks(np.arange(0, int(ax.get_xlim()[1]) + 1))
        ax.set_yticks(np.arange(0, int(ax.get_ylim()[1]) + 1))

    # Create combined plot in the last subplot
    if all_complete_data:
        ax_combined = axes[-1]

        # Combine all data
        combined_data = pd.concat(all_complete_data, ignore_index=True)

        # Plot each F1_condition with different colors
        for i, f1_condition in enumerate(f1_conditions):
            condition_subset = combined_data[combined_data["f1_condition"] == f1_condition]
            if condition_subset.empty:
                continue

            color = colors[i % len(colors)]
            x = condition_subset["training"]
            y = condition_subset["test"]

            ax_combined.scatter(
                x,
                y,
                alpha=0.7,
                s=60,
                edgecolors="black",
                linewidth=0.5,
                color=color,
                label=f"{f1_condition} (n={len(condition_subset)})",
            )

            # Add trend line for each condition if there's variation
            if len(x) > 1 and x.std() > 0 and y.std() > 0:
                z = np.polyfit(x, y, 1)
                fit_line = np.poly1d(z)
                ax_combined.plot(x, fit_line(x), color=color, alpha=0.8, linewidth=2, linestyle="--")

        ax_combined.set_xlabel("Pulled Events - Training Ball", fontsize=11)
        ax_combined.set_ylabel("Pulled Events - Test Ball", fontsize=11)
        ax_combined.set_title("Combined: All F1_conditions", fontsize=12, fontweight="bold")
        ax_combined.grid(True, alpha=0.3)
        ax_combined.legend()

        # Set limits and ticks for combined plot
        all_x = combined_data["training"]
        all_y = combined_data["test"]
        if len(all_x) > 0 and len(all_y) > 0:
            ax_combined.set_xlim(max(0, all_x.min() - 0.5), all_x.max() + 0.5)
            ax_combined.set_ylim(max(0, all_y.min() - 0.5), all_y.max() + 0.5)
            ax_combined.set_xticks(np.arange(0, int(ax_combined.get_xlim()[1]) + 1))
            ax_combined.set_yticks(np.arange(0, int(ax_combined.get_ylim()[1]) + 1))

    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "training_vs_test_pulled_events_by_f1condition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print comparison statistics
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS F1_CONDITIONS - PULLED EVENTS")
    print("=" * 60)

    for f1_condition, result in results.items():
        print(f"\nF1_condition: {f1_condition}")
        print(f"  Number of flies: {result['n_flies']}")
        if not np.isnan(result["correlation"]):
            print(f"  Correlation: {result['correlation']:.3f} (p = {result['p_value']:.3f})")
            print(f"  Slope: {result['slope']:.3f}")
            print(f"  Training pulled events - Mean: {result['x_data'].mean():.2f} ± {result['x_data'].std():.2f}")
            print(f"  Test pulled events - Mean: {result['y_data'].mean():.2f} ± {result['y_data'].std():.2f}")
            if result["x_data"].mean() > 0:
                print(f"  Test/Training ratio: {result['y_data'].mean() / result['x_data'].mean():.3f}")
            else:
                print("  Test/Training ratio: undefined (training mean = 0)")
        else:
            print("  Insufficient data/variation for correlation analysis")
            print(f"  Training pulled events - Mean: {result['x_data'].mean():.2f} ± {result['x_data'].std():.2f}")
            print(f"  Test pulled events - Mean: {result['y_data'].mean():.2f} ± {result['y_data'].std():.2f}")

    # Statistical comparison between conditions
    if len(results) >= 2:
        print(f"\n" + "-" * 40)
        print("STATISTICAL COMPARISON BETWEEN CONDITIONS")
        print("-" * 40)

        conditions = list(results.keys())
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]

                print(f"\n{cond1} vs {cond2}:")

                if not np.isnan(results[cond1]["correlation"]) and not np.isnan(results[cond2]["correlation"]):
                    print(
                        f"  Correlation difference: {results[cond1]['correlation'] - results[cond2]['correlation']:.3f}"
                    )
                    print(f"  Slope difference: {results[cond1]['slope'] - results[cond2]['slope']:.3f}")

                # Test for difference in means between test ball pulled events
                from scipy.stats import ttest_ind, mannwhitneyu

                # Use Mann-Whitney U test for count data (non-parametric)
                try:
                    u_stat, u_p_val = mannwhitneyu(
                        results[cond1]["y_data"], results[cond2]["y_data"], alternative="two-sided"
                    )
                    print(f"  Test pulled events mean difference (Mann-Whitney U): U = {u_stat:.1f}, p = {u_p_val:.3f}")
                except ValueError as e:
                    print(f"  Could not perform Mann-Whitney U test: {e}")

                # Also perform t-test for comparison
                try:
                    t_stat, t_p_val = ttest_ind(results[cond1]["y_data"], results[cond2]["y_data"])
                    print(f"  Test pulled events mean difference (t-test): t = {t_stat:.3f}, p = {t_p_val:.3f}")
                except:
                    print("  Could not perform t-test")

    # Additional insights for pulled events
    print(f"\n" + "-" * 40)
    print("PULLED EVENTS INSIGHTS")
    print("-" * 40)

    for f1_condition, result in results.items():
        x_data = result["x_data"]
        y_data = result["y_data"]

        print(f"\n{f1_condition}:")
        print(f"  Training pulled events range: {x_data.min():.0f} - {x_data.max():.0f}")
        print(f"  Test pulled events range: {y_data.min():.0f} - {y_data.max():.0f}")

        # Count flies with any pulling events
        any_training_pulling = sum(x_data > 0)
        any_test_pulling = sum(y_data > 0)
        print(
            f"  Flies with any training pulling events: {any_training_pulling}/{len(x_data)} ({100*any_training_pulling/len(x_data):.1f}%)"
        )
        print(
            f"  Flies with any test pulling events: {any_test_pulling}/{len(y_data)} ({100*any_test_pulling/len(y_data):.1f}%)"
        )

        # Count flies that show pulling in both phases
        consistent_pullers = sum((x_data > 0) & (y_data > 0))
        print(
            f"  Consistent pullers (any pulling in both): {consistent_pullers}/{len(x_data)} ({100*consistent_pullers/len(x_data):.1f}%)"
        )

        # Count flies with high pulling (>2 events)
        high_training_pulling = sum(x_data > 2)
        high_test_pulling = sum(y_data > 2)
        print(
            f"  Flies with high training pulling (>2 events): {high_training_pulling}/{len(x_data)} ({100*high_training_pulling/len(x_data):.1f}%)"
        )
        print(
            f"  Flies with high test pulling (>2 events): {high_test_pulling}/{len(y_data)} ({100*high_test_pulling/len(y_data):.1f}%)"
        )

    plt.show()
    return results


if __name__ == "__main__":
    data = main()
