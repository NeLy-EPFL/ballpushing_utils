from utils_behavior import HoloviewsTemplates, Processing
import pandas as pd
import holoviews as hv
import panel as pn
import Config
from pathlib import Path
import re

from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import datetime
from icecream import ic
import pandas as pd
import colorcet as cc
from bokeh.palettes import Category10
from bokeh.palettes import all_palettes
from bokeh.io.export import export_svgs

hv.extension("bokeh")
pn.extension()

# sns.set_theme()
rg = np.random.default_rng()

dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# Configuration section
OUTPUT_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics/All_metrics_bysplit_BS_CI_99/"
)  # Base directory for saving plots
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# dataset = Config.map_split_registry(dataset)

controls = ["Empty-Split", "Empty-Gal4", "TNTxPR"]

# Convert boolean metrics to numeric (1 for True, 0 for False)
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)

# Configure which are the metrics and what are the metadata

metric_names = [
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
    "major_event_first",
    # "insight_effect",
    # "insight_effect_log",
    "cumulated_breaks_duration",
    "chamber_time",
    "chamber_ratio",
    "pushed",
    "pulled",
    "pulling_ratio",
    "success_direction",
    "interaction_proportion",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    "exit_time",
    "chamber_exit_time",
    "number_of_pauses",
    "total_pause_duration",
    "learning_slope",
    "learning_slope_r2",
    "logistic_L",
    "logistic_k",
    "logistic_t0",
    "logistic_r2",
    "avg_displacement_after_success",
    "avg_displacement_after_failure",
    "influence_ratio",
    "normalized_velocity",
    "velocity_during_interactions",
    "velocity_trend",
]
# Define metadata columns as any column not in metric_names
metadata_names = [col for col in dataset.columns if col not in metric_names]

print(f"Metadata columns: {metadata_names}")
print(f"Metric columns: {metric_names}")

# Define metric sublists
events_metrics = [
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
    "major_event_first",
]

activity_metrics = [
    "velocity_during_interactions",
    "velocity_trend",
    "normalized_velocity",
    "chamber_exit_time",
    "interaction_proportion",
    "interaction_persistence",
]

learning_metrics = [
    "learning_slope",
    "learning_slope_r2",
    "logistic_L",
    "logistic_k",
    "logistic_t0",
    "logistic_r2",
]

insight_metrics = [
    # "insight_effect",
    # "insight_effect_log",
    "avg_displacement_after_success",
    "avg_displacement_after_failure",
    "influence_ratio",
]

behavior_strategies = [
    "pushed",
    "pulled",
    "pulling_ratio",
    # "success_direction",
    "distance_moved",
    "distance_ratio",
    "total_pause_duration",
    "number_of_pauses",
]

# Combine all sublists into a dictionary
metric_groups = {
    "Events Metrics": events_metrics,
    "Activity Metrics": activity_metrics,
    "Learning Metrics": learning_metrics,
    "Insight Metrics": insight_metrics,
    "Behavior Strategies": behavior_strategies,
}

# Set metadata names
metric_names = sum(metric_groups.values(), [])  # Flatten all metric sublists
metadata_names = [col for col in dataset.columns if col not in metric_names]


# For each column, handle missing values gracefully

# For max event, first significant event, major event final event, if missing, set to -1
dataset["max_event"].fillna(-1, inplace=True)
dataset["first_significant_event"].fillna(-1, inplace=True)
dataset["major_event"].fillna(-1, inplace=True)
dataset["final_event"].fillna(-1, inplace=True)

# For max event time, first significant event time, major event final event time, if missing, set to 3600
dataset["max_event_time"].fillna(3600, inplace=True)
dataset["first_significant_event_time"].fillna(3600, inplace=True)
dataset["first_major_event_time"].fillna(3600, inplace=True)
dataset["final_event_time"].fillna(3600, inplace=True)

# Remove columns insight_effect, insight_effect_log, exit_time
dataset.drop(columns=["insight_effect", "insight_effect_log", "exit_time"], inplace=True)

# for pulling_ratio, avg_displacement_after_success, avg_displacement_before_success, and influence_ratio, if missing set to 0
dataset["pulling_ratio"].fillna(0, inplace=True)
dataset["avg_displacement_after_success"].fillna(0, inplace=True)
dataset["avg_displacement_after_failure"].fillna(0, inplace=True)
dataset["influence_ratio"].fillna(0, inplace=True)

# Check which metrics have NA values
na_metrics = dataset[metric_names].isna().sum()
print("Metrics with NA values:")
print(na_metrics[na_metrics > 0])


def jitterboxplot(data, x, y, hue=None, palette="Set2", title=None, figsize=(10, 6), ax=None):
    """
    Generates a single jitterboxplot combining a horizontal boxplot and a stripplot.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        x (str): The name of the column for the x-axis (numeric values).
        y (str): The name of the column for the y-axis (categorical values).
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plot. Default is "Set2".
        title (str, optional): Title of the plot. Default is None.
        figsize (tuple, optional): Size of the figure. Default is (10, 6).
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.

    Returns:
        None: Displays the plot.
    """
    # Drop rows where x or y is NaN to avoid issues during grouping
    data = data.dropna(subset=[x, y])

    # Sort the y-axis categories by the median value of x
    sorted_categories = data.groupby(y, observed=True)[x].median().sort_values(ascending=False).index
    data[y] = pd.Categorical(data[y], categories=sorted_categories, ordered=True)

    # Sort the DataFrame based on the new categorical order of y
    data = data.sort_values(by=[y], key=lambda col: col.cat.codes)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Create the boxplot
    sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette, showfliers=False, width=0.6, ax=ax)

    # Overlay the stripplot for jitter
    sns.stripplot(data=data, x=x, y=y, dodge=True, alpha=0.6, jitter=True, color="black", size=2, ax=ax)

    # Adjust legend to avoid duplication
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[: len(data[hue].unique())], labels[: len(data[hue].unique())], title=hue)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()


def create_layout(data, metrics, y, hue=None, palette="Set2", figsize=(10, 6), ncols=2):
    """
    Creates a grid layout of jitterboxplots for a given list of metrics.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (categorical values).
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).
        ncols (int, optional): Number of columns in the grid layout. Default is 2.

    Returns:
        None: Displays the grid layout of plots.
    """
    nrows = -(-len(metrics) // ncols)  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, metric in enumerate(metrics):
        print(f"Creating jitterboxplot for metric: {metric}")
        jitterboxplot(
            data=data, x=metric, y=y, hue=hue, palette=palette, title=f"Jitterboxplot of {metric} by {y}", ax=axes[i]
        )

    # Hide any unused subplots
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.show()


def generate_layouts_by_brain_region(
    data, brain_regions, metric_groups, y, hue=None, palette="Set2", figsize=(10, 6), ncols=2
):
    """
    Generates layouts for all combinations of brain regions and metric groups.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        brain_regions (list): List of unique brain regions.
        metric_groups (dict): Dictionary of metric groups.
        y (str): The name of the column for the y-axis (categorical values).
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).
        ncols (int, optional): Number of columns in the grid layout. Default is 2.

    Returns:
        None
    """
    control_data = data[data["Brain region"] == "Control"]

    for brain_region in brain_regions:
        # Filter data for the current brain region
        region_data = data[data["Brain region"] == brain_region]

        # Combine control data with the current brain region
        combined_data = pd.concat([control_data, region_data], axis=0).reset_index(drop=True)

        # Create a directory for the current brain region
        region_dir = OUTPUT_DIR / brain_region
        region_dir.mkdir(parents=True, exist_ok=True)

        for group_name, metrics in metric_groups.items():
            print(f"Generating layout for Brain Region: {brain_region}, Metric Group: {group_name}")

            # Create the layout for the current metric group
            create_layout(
                data=combined_data,
                metrics=metrics,
                y=y,
                hue=hue,
                palette=palette,
                figsize=figsize,
                ncols=ncols,
            )

            # Save the plot to the corresponding directory
            output_path = region_dir / f"{group_name.replace(' ', '_')}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close the plot to free memory


def generate_jitterboxplots_for_all_metrics(
    data,
    metrics,
    y,
    hue=None,
    palette="Set2",
    figsize=(15, 10),
    output_dir="all_metrics_plots",
    control_name=None,
    ax=None,
    stats_df=None,
):
    """
    Generates a jitterboxplot for each metric using the entire dataset.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (categorical values).
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (15, 10).
        output_dir (str or Path, optional): Directory to save the plots. Default is "all_metrics_plots".
        control_name (str, optional): Name of the control group.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        stats_df (pd.DataFrame, optional): DataFrame with columns ["Nickname", "Metric", "significant"] for annotation.

    Returns:
        None
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(f"Generating jitterboxplot for metric: {metric}")

        plot_data = data.dropna(subset=[metric, y])

        # Sort the y-axis categories by the median value of x
        sorted_categories = plot_data.groupby(y, observed=True)[metric].median().sort_values(ascending=False).index
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_categories, ordered=True)
        plot_data = plot_data.sort_values(by=[y], key=lambda col: col.cat.codes)

        fig, ax = plt.subplots(figsize=figsize)

        # Add gray background to every other row
        for idx, group in enumerate(plot_data[y].cat.categories):
            if idx % 2 == 1:
                ax.axhspan(idx - 0.5, idx + 0.5, color="lightgray", alpha=0.2, zorder=0)

        # Calculate and plot control group's bootstrapped CI as a vertical band
        control_vals = plot_data[plot_data[y] == control_name][metric].dropna()
        if len(control_vals) > 0:
            boot_means = [
                np.mean(np.random.choice(control_vals, size=len(control_vals), replace=True)) for _ in range(1000)
            ]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            ax.axvspan(ci_low, ci_high, color="red", alpha=0.15, label="Control 95% CI")

        # --- Draw boxplots for all groups ---
        non_control = plot_data[plot_data[y] != control_name]
        if not non_control.empty:
            sns.boxplot(
                data=non_control,
                x=metric,
                y=y,
                hue=None,
                palette=None,
                showfliers=False,
                width=0.6,
                ax=ax,
                boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
            )
        control = plot_data[plot_data[y] == control_name]
        if not control.empty:
            sns.boxplot(
                data=control,
                x=metric,
                y=y,
                hue=None,
                palette=None,
                showfliers=False,
                width=0.6,
                ax=ax,
                boxprops=dict(facecolor="none", edgecolor="red", linewidth=1, linestyle="--"),
                medianprops=dict(color="red", linewidth=1.5),
                whiskerprops=dict(color="red", linewidth=1, linestyle="--"),
                capprops=dict(color="red", linewidth=1),
            )

        # Overlay stripplot for jitter
        sns.stripplot(
            data=plot_data, x=metric, y=y, dodge=False, alpha=0.7, jitter=True, hue=hue, palette=palette, size=4, ax=ax
        )

        # Calculate and plot control group's bootstrapped CI as a vertical band
        control_vals = plot_data[plot_data[y] == control_name][metric].dropna()
        if len(control_vals) > 0:
            boot_means = [
                np.mean(np.random.choice(control_vals, size=len(control_vals), replace=True)) for _ in range(1000)
            ]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            ax.axvspan(ci_low, ci_high, color="red", alpha=0.15, label="Control 95% CI")

        # Set xlim to 0.99 quantile of the metric
        x_max = plot_data[metric].quantile(0.99)
        ax.set_xlim(left=None, right=x_max)

        # --- Add asterisks for significant Nicknames using stats_df ---
        significant_groups = set()
        if stats_df is not None:
            sig_rows = stats_df[(stats_df["Metric"] == metric) & (stats_df["significant"])]
            significant_groups = set(sig_rows["Nickname"].astype(str))

        # After plotting, add red asterisks next to significant groups
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        for idx, label in enumerate(yticklabels):
            if label in significant_groups:
                yticklocs = ax.get_yticks()
                ax.text(
                    x=ax.get_xlim()[1],  # Far right of the plot
                    y=yticklocs[idx],
                    s="*",
                    color="red",
                    fontsize=18,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        # Font and labels
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel(metric, fontsize=10)
        plt.ylabel(y, fontsize=10)
        plt.title(f"Jitterboxplot of {metric} by {y}", fontsize=12)
        if hue:
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), title=hue, fontsize=8)
        else:
            handles = [
                Patch(facecolor="none", edgecolor="red", linestyle="--", label="Control Box"),
            ]
            ax.legend(handles=handles, fontsize=8)
        plt.tight_layout()

        output_path = output_dir / f"{metric.replace(' ', '_')}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        output_eps = output_dir / f"{metric.replace(' ', '_')}.eps"
        plt.savefig(output_eps, dpi=300, bbox_inches="tight")

        plt.close(fig)


# Now let's create one example layout with the first metric group
group = "Events Metrics"

dataset = Config.cleanup_data(dataset)

# Get a brain region and

Control_data = dataset[dataset["Brain region"] == "Control"]

ex_region = dataset[dataset["Brain region"] == "MB"]

Combined = pd.concat([Control_data, ex_region], axis=0).reset_index(drop=True)

# Check for duplicate indices (optional for debugging)
print("Checking for duplicate indices...")
print(Combined.index.duplicated().sum(), "duplicate indices found.")

# Save as pdf
# plt.savefig("exLayout.pdf", dpi=300, bbox_inches="tight")


brain_regions = dataset["Brain region"].unique()  # Get unique brain regions

# Generate jitterboxplots for all metrics


SplitLines = dataset[dataset["Split"] == "y"]
NoSplit = dataset[dataset["Split"] == "n"]
Mutants = dataset[dataset["Split"] == "m"]


def nickname_control_map(split):
    if split == "y":
        return "Empty-Split"
    elif split == "n":
        return "Empty-Gal4"
    elif split == "m":
        return "TNTxPR"
    else:
        return None


def mannwhitney_nickname_vs_control_table(
    data, metrics, y="Nickname", split_col="Split", output_csv="nickname_vs_control_stats.csv"
):
    results = []
    nicknames = data[y].unique()
    for metric in metrics:
        metric_results = []
        for nickname in nicknames:
            split_val = data.loc[data[y] == nickname, split_col].iloc[0]
            control_name = nickname_control_map(split_val)
            if control_name is None or nickname == control_name:
                continue
            group_vals = data.loc[data[y] == nickname, metric].dropna()
            control_vals = data.loc[data[y] == control_name, metric].dropna()
            if len(group_vals) > 0 and len(control_vals) > 0:
                try:
                    stat, pval = mannwhitneyu(group_vals, control_vals, alternative="two-sided")
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0
            metric_results.append(
                {
                    "Nickname": nickname,
                    "Split": split_val,
                    "Control": control_name,
                    "Metric": metric,
                    "pval": pval,
                }
            )
        # Holm correction for this metric across all Nicknames
        pvals = [r["pval"] for r in metric_results]
        if pvals:  # Only correct if there are tests
            reject, pvals_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            for i, r in enumerate(metric_results):
                r["pval_holm"] = pvals_holm[i]
                r["significant"] = reject[i]
        results.extend(metric_results)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return df


def bootstrapped_ci_stats_df(data, metrics, y="Nickname", split_col="Split", n_boot=1000, alpha=0.05):
    """
    Returns a DataFrame with columns: Nickname, Metric, significant (bool), pval_holm (set to np.nan), etc.
    Each Nickname is compared to its corresponding control (based on Split value).
    """
    results = []
    nicknames = data[y].unique()
    for metric in metrics:
        plot_data = data.dropna(subset=[metric, y, split_col])
        for nickname in nicknames:
            split_val = plot_data.loc[plot_data[y] == nickname, split_col].iloc[0]
            control_name = nickname_control_map(split_val)
            if control_name is None or nickname == control_name:
                continue
            group_vals = plot_data[plot_data[y] == nickname][metric].dropna()
            control_vals = plot_data[plot_data[y] == control_name][metric].dropna()
            if len(group_vals) == 0 or len(control_vals) == 0:
                continue
            control_boot = [
                np.mean(np.random.choice(control_vals, size=len(control_vals), replace=True)) for _ in range(n_boot)
            ]
            group_boot = [
                np.mean(np.random.choice(group_vals, size=len(group_vals), replace=True)) for _ in range(n_boot)
            ]
            ci_low, ci_high = np.percentile(control_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            g_low, g_high = np.percentile(group_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            significant = (g_high < ci_low) or (g_low > ci_high)
            results.append(
                {
                    "Nickname": nickname,
                    "Metric": metric,
                    "significant": significant,
                    "pval_holm": np.nan,  # Not used here
                }
            )
    return pd.DataFrame(results)


# # Example usage
# stats_df = mannwhitney_nickname_vs_control_table(
#     dataset, metric_names, y="Nickname", split_col="Split", output_csv="nickname_vs_control_stats.csv"
# )

# stats_df = mannwhitney_nickname_vs_control_table(
#     dataset, metric_names, y="Nickname", split_col="Split", output_csv="nickname_vs_control_stats.csv"
# )

stats_df = bootstrapped_ci_stats_df(dataset, metric_names, y="Nickname", n_boot=1000, alpha=0.01)

### Grouping by control

### Example case

# example_metric = ["nb_events"]
# generate_jitterboxplots_for_all_metrics(
#     data=SplitLines,
#     metrics=example_metric,
#     y="Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     figsize=(10, 30),
#     output_dir=OUTPUT_DIR,
#     control_name="Empty-Split",
# )

generate_jitterboxplots_for_all_metrics(
    data=SplitLines,
    metrics=metric_names,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 30),
    output_dir=OUTPUT_DIR / "split",
    control_name="Empty-Split",
    stats_df=stats_df,
)

generate_jitterboxplots_for_all_metrics(
    data=NoSplit,
    metrics=metric_names,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 30),
    output_dir=OUTPUT_DIR / "nosplit",
    control_name="Empty-Gal4",
    stats_df=stats_df,
)

generate_jitterboxplots_for_all_metrics(
    data=Mutants,
    metrics=metric_names,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 30),
    output_dir=OUTPUT_DIR / "mutant",
    control_name="TNTxPR",  # Control name for mutants
    stats_df=stats_df,
)
