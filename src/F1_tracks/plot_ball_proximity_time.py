#!/usr/bin/env python3
"""
Calculate and plot the proportion of time flies spend near the ball.
Compares control vs experimental conditions for both F1_condition and Pretraining groupings.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy import stats
import argparse


def filter_data_from_first_movement(df, x_col, fly_col="fly", threshold=100, n_avg=100):
    """
    Filter data to keep only points after fly first moves >threshold from starting position.

    Args:
        df: DataFrame with fly tracking data
        x_col: Column name for x coordinates
        fly_col: Column name for fly identifier
        threshold: Distance threshold in pixels (default: 100)
        n_avg: Number of initial points to average for starting position (default: 100)

    Returns:
        Filtered DataFrame
    """
    filtered_dfs = []

    for fly_id in df[fly_col].unique():
        fly_data = df[df[fly_col] == fly_id].copy()

        # Sort by frame or index to ensure temporal order
        if "frame" in fly_data.columns:
            fly_data = fly_data.sort_values("frame")
        else:
            fly_data = fly_data.sort_index()

        # Calculate starting position (average of first n_avg points)
        n_use = min(n_avg, len(fly_data))
        if n_use == 0:
            continue

        x_start = fly_data[x_col].iloc[:n_use].mean()

        # Calculate distance from start
        fly_data["distance_from_start"] = np.abs(fly_data[x_col] - x_start)

        # Find first time distance exceeds threshold
        threshold_mask = fly_data["distance_from_start"] > threshold

        if threshold_mask.any():
            first_threshold_idx = threshold_mask.idxmax()
            # Keep all data from this point onwards
            fly_data_filtered = fly_data.loc[first_threshold_idx:]
            filtered_dfs.append(fly_data_filtered)

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_ball_proximity_proportion(df, proximity_threshold=70):
    """
    Calculate the proportion of time each fly spends near the ball.

    Args:
        df: DataFrame with fly and ball position data
        proximity_threshold: Distance threshold for proximity in pixels (default: 70)

    Returns:
        DataFrame with fly_id, condition, and proximity_proportion
    """
    results = []

    for fly_id in df["fly"].unique():
        fly_data = df[df["fly"] == fly_id].copy()

        if len(fly_data) == 0:
            continue

        # Get condition information
        f1_condition = fly_data["F1_condition"].iloc[0] if "F1_condition" in fly_data.columns else None
        pretraining = fly_data["Pretraining"].iloc[0] if "Pretraining" in fly_data.columns else None

        # Determine which ball to use based on condition
        # Control flies: ball_0, Experimental flies: ball_1
        if f1_condition == "control" or pretraining == "n":
            ball_x_col = "x_centre_ball_0"
            ball_y_col = "y_centre_ball_0"
        else:
            ball_x_col = "x_centre_ball_1"
            ball_y_col = "y_centre_ball_1"

        # Check if ball columns exist
        if ball_x_col not in fly_data.columns or ball_y_col not in fly_data.columns:
            continue

        # Get fly and ball positions
        fly_x = fly_data["x_thorax_fly_0"].values
        fly_y = fly_data["y_thorax_fly_0"].values
        ball_x = fly_data[ball_x_col].values
        ball_y = fly_data[ball_y_col].values

        # Calculate distance to ball (Euclidean distance)
        distance_to_ball = np.sqrt((fly_x - ball_x) ** 2 + (fly_y - ball_y) ** 2)

        # Calculate proportion of time near ball
        near_ball = distance_to_ball < proximity_threshold
        proportion_near = near_ball.sum() / len(near_ball) if len(near_ball) > 0 else 0

        results.append(
            {
                "fly": fly_id,
                "F1_condition": f1_condition,
                "Pretraining": pretraining,
                "proximity_proportion": proportion_near,
                "n_timepoints": len(fly_data),
                "n_near_ball": near_ball.sum(),
            }
        )

    return pd.DataFrame(results)


def plot_proximity_comparison(data, group_by="F1_condition", control_value="control", output_path=None):
    """
    Create boxplot with scatter overlay comparing control vs experimental conditions.
    Style matches run_mannwhitney_f1_metrics.py

    Args:
        data: DataFrame with proximity proportions
        group_by: Column to group by ('F1_condition' or 'Pretraining')
        control_value: Value that represents control condition
        output_path: Path to save the plot
    """
    import matplotlib

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Filter out rows with missing group_by values
    data_clean = data[data[group_by].notna()].copy()

    if len(data_clean) == 0:
        print(f"   No data to plot for {group_by}")
        return

    # Get unique conditions and sort
    conditions = sorted(data_clean[group_by].unique())

    # Put control first
    if control_value in conditions:
        conditions.remove(control_value)
        conditions = [control_value] + conditions

    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(4, 6))

    # Define colors matching the reference script
    pretraining_colors = {
        "control": "#ff7f0e",  # Orange for control
        "pretrained": "#1f77b4",  # Blue for pretrained
        "pretrained_unlocked": "#2ca02c",  # Green for pretrained_unlocked
        "n": "#ff7f0e",  # Orange for no pretraining (control)
        "y": "#1f77b4",  # Blue for pretrained
    }

    final_colors = [pretraining_colors.get(c, "#2ca02c") for c in conditions]

    # Prepare data for boxplot
    box_data = [
        data_clean[data_clean[group_by] == condition]["proximity_proportion"].values for condition in conditions
    ]

    # Create boxplot
    bp = ax.boxplot(
        box_data,
        positions=range(len(conditions)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], final_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    # Overlay scatter points with jitter
    np.random.seed(42)
    for i, condition in enumerate(conditions):
        condition_data = data_clean[data_clean[group_by] == condition]["proximity_proportion"].values
        # Add jitter to x positions
        x_positions = np.random.normal(i, 0.06, size=len(condition_data))
        ax.scatter(
            x_positions,
            condition_data,
            alpha=0.5,
            s=30,
            color=final_colors[i],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    # Set labels and title
    ax.set_xticks(range(len(conditions)))

    # Map condition labels
    label_map = {
        "y": "Pretrained",
        "n": "Ctrl",
        "control": "Control",
        "pretrained": "Pretrained",
        "pretrained_unlocked": "Pretrained\nUnlocked",
    }
    x_labels = [label_map.get(c, c) for c in conditions]
    ax.set_xticklabels(x_labels, fontsize=11)

    ax.set_ylabel("Proportion of Time Near Ball", fontsize=12)

    # Simpler title
    metric_name = "Ball Proximity Time"
    ax.set_title(f"{metric_name}", fontsize=13, fontweight="bold", pad=10)

    # Add horizontal grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add sample sizes below x-axis
    y_min = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - y_min
    for i, condition in enumerate(conditions):
        n = len(data_clean[data_clean[group_by] == condition])
        ax.text(i, y_min - 0.08 * y_range, f"n={n}", ha="center", va="top", fontsize=9, color="gray")

    # Perform statistical tests
    print(f"\n   Statistical Analysis for {group_by}:")
    print(f"   {'-'*50}")

    # Overall test (Kruskal-Wallis if more than 2 groups, Mann-Whitney if 2)
    groups = [data_clean[data_clean[group_by] == condition]["proximity_proportion"].values for condition in conditions]

    if len(conditions) > 2:
        stat, p_value = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis"
    else:
        stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        test_name = "Mann-Whitney U"

    print(f"   {test_name}: statistic={stat:.4f}, p={p_value:.4e}")

    # Add significance annotation to plot if 2 groups
    if len(conditions) == 2:
        # Add significance bar
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        # Determine significance level
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        # Draw significance bar
        x1, x2 = 0, 1
        y_bar = y_max + 0.02 * y_range
        ax.plot([x1, x2], [y_bar, y_bar], "k-", linewidth=1.5)
        ax.plot([x1, x1], [y_bar - 0.01 * y_range, y_bar], "k-", linewidth=1.5)
        ax.plot([x2, x2], [y_bar - 0.01 * y_range, y_bar], "k-", linewidth=1.5)
        ax.text(
            (x1 + x2) / 2, y_bar + 0.01 * y_range, sig_text, ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    # Pairwise comparisons with control
    if control_value in conditions and len(conditions) > 2:
        control_data = data_clean[data_clean[group_by] == control_value]["proximity_proportion"].values
        print(f"\n   Pairwise Mann-Whitney U tests vs {control_value}:")

        for condition in conditions:
            if condition != control_value:
                exp_data = data_clean[data_clean[group_by] == condition]["proximity_proportion"].values
                stat, p_value = stats.mannwhitneyu(control_data, exp_data, alternative="two-sided")

                # Calculate medians
                control_median = np.median(control_data)
                exp_median = np.median(exp_data)

                # Determine significance
                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                print(f"     {control_value} vs {condition}: U={stat:.1f}, p={p_value:.4e} {sig}")
                print(f"       Medians: {control_median:.4f} vs {exp_median:.4f}")

    # Print descriptive statistics
    print(f"\n   Descriptive Statistics:")
    print(f"   {'-'*50}")
    for condition in conditions:
        condition_data = data_clean[data_clean[group_by] == condition]["proximity_proportion"].values
        print(
            f"   {condition}: n={len(condition_data)}, "
            f"median={np.median(condition_data):.4f}, "
            f"mean={np.mean(condition_data):.4f} ± {np.std(condition_data):.4f}"
        )

    plt.tight_layout()

    if output_path:
        # Save as PDF for vector graphics
        pdf_path = str(output_path).replace(".png", ".pdf")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"\n   Saved plot to: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate and plot ball proximity time")
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=70,
        help="Distance threshold for proximity in pixels (default: 70)",
    )

    args = parser.parse_args()

    # Paths
    dataset_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251121_17_summary_F1_New_Data/fly_positions/pooled_fly_positions.feather"
    )
    output_dir = Path(__file__).parent

    print("=" * 70)
    print("BALL PROXIMITY TIME ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_feather(dataset_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Number of unique flies: {df['fly'].nunique()}")

    # Find position columns
    x_col = None
    y_col = None
    possible_x_cols = ["x_thorax_fly_0", "x_thorax", "x_thorax_skeleton"]
    possible_y_cols = ["y_thorax_fly_0", "y_thorax", "y_thorax_skeleton"]

    for col in possible_x_cols:
        if col in df.columns:
            x_col = col
            break

    for col in possible_y_cols:
        if col in df.columns:
            y_col = col
            break

    if x_col is None or y_col is None:
        print(f"   ERROR: Could not find position columns!")
        return

    print(f"   Using position columns: x_col={x_col}, y_col={y_col}")

    # Filter data: keep only from first time fly moves >100px from start
    print("\n2. Filtering data from first movement >100px from start...")
    df_filtered = filter_data_from_first_movement(df, x_col, fly_col="fly", threshold=100, n_avg=100)
    print(f"   Dataset shape after movement filter: {df_filtered.shape}")
    print(f"   Number of unique flies after filter: {df_filtered['fly'].nunique()}")

    if len(df_filtered) == 0:
        print("   ERROR: No data remaining after filter!")
        return

    # Calculate ball proximity proportions
    print(f"\n3. Calculating ball proximity proportions (threshold={args.proximity_threshold}px)...")
    proximity_data = calculate_ball_proximity_proportion(df_filtered, proximity_threshold=args.proximity_threshold)
    print(f"   Calculated proximity for {len(proximity_data)} flies")

    if len(proximity_data) == 0:
        print("   ERROR: No proximity data calculated!")
        return

    # Save results to CSV
    csv_path = output_dir / f"ball_proximity_proportions_threshold{args.proximity_threshold}.csv"
    proximity_data.to_csv(csv_path, index=False)
    print(f"   Saved results to: {csv_path}")

    # Plot by F1_condition
    print("\n4. Plotting by F1_condition...")
    output_path_f1 = output_dir / f"ball_proximity_by_F1_condition_threshold{args.proximity_threshold}.png"
    plot_proximity_comparison(
        proximity_data, group_by="F1_condition", control_value="control", output_path=output_path_f1
    )

    # Plot by Pretraining
    print("\n5. Plotting by Pretraining...")
    output_path_pretraining = output_dir / f"ball_proximity_by_Pretraining_threshold{args.proximity_threshold}.png"
    plot_proximity_comparison(
        proximity_data, group_by="Pretraining", control_value="n", output_path=output_path_pretraining
    )

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
