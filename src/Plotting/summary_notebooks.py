from utils_behavior import HoloviewsTemplates, Processing
import pandas as pd
import holoviews as hv
import panel as pn
import Config
from pathlib import Path
import re


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from icecream import ic
import pandas as pd
import colorcet as cc
from bokeh.palettes import Category10
from bokeh.palettes import all_palettes
from bokeh.io.export import export_svgs

hv.extension("bokeh")
pn.extension()


rg = np.random.default_rng()

dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# Configuration section
OUTPUT_DIR = Path("/home/durrieu/ballpushing_utils/outputs")  # Base directory for saving plots
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
    "major_event",
    "major_event_time",
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
    "major_event",
    "major_event_time",
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
dataset["major_event_time"].fillna(3600, inplace=True)
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


# Example usage of the jitterboxplot function
jitterboxplot(
    data=dataset,
    x="nb_events",  # Numeric column
    y="Simplified Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    title="Jitterboxplot of Number of Events by Brain Region",
)

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
    data, metrics, y, hue=None, palette="Set2", figsize=(15, 10), output_dir="all_metrics_plots"
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

    Returns:
        None
    """
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(f"Generating jitterboxplot for metric: {metric}")

        # Drop rows where the metric or y is NaN
        plot_data = data.dropna(subset=[metric, y])

        # Explicitly create a new figure with the specified figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Create the jitterboxplot
        jitterboxplot(
            data=plot_data,
            x=metric,
            y=y,
            hue=hue,
            palette=palette,
            title=f"Jitterboxplot of {metric} by {y}",
            ax=ax,  # Pass the created axis
        )

        # Make all font smaller
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel(metric, fontsize=10)
        plt.ylabel(y, fontsize=10)
        plt.title(f"Jitterboxplot of {metric} by {y}", fontsize=12)
        plt.legend(title=hue, fontsize=8) if hue else None
        plt.tight_layout()

        # Save the plot
        output_path = output_dir / f"{metric.replace(' ', '_')}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory


# Example usage of the jitterboxplot function
explot = jitterboxplot(
    data=Combined,
    x="nb_events",  # Numeric column
    y="Simplified Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    title="Jitterboxplot of Number of Events by Brain Region",
)

plt.savefig("jitterboxplot.pdf", dpi=300, bbox_inches="tight")

explot = create_layout(
    data=Combined,
    metrics=metric_groups[group],
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 6),
)

# Save as pdf
plt.savefig("exLayout.pdf", dpi=300, bbox_inches="tight")

brain_regions = dataset["Brain region"].unique()  # Get unique brain regions
generate_layouts_by_brain_region(
    data=dataset,
    brain_regions=brain_regions,
    metric_groups=metric_groups,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 6),
    ncols=3,
)

# Generate jitterboxplots for all metrics

generate_jitterboxplots_for_all_metrics(
    data=dataset,
    metrics=metric_names,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 30),
    output_dir=OUTPUT_DIR / "All_Metrics",
)

# Load the PCA results

pca_results = pd.read_feather("/home/durrieu/ballpushing_utils/outputs/pca_with_metadata.feather")


def generate_pca_jitterboxplots_by_brain_region(
    pca_data, brain_regions, components, y, output_dir, hue=None, palette="Set2", figsize=(10, 6), ncols=3
):
    """
    Generates jitterboxplots of PCA components by Nickname for each brain region.

    Parameters:
        pca_data (pd.DataFrame): The PCA dataset containing components and metadata.
        brain_regions (list): List of unique brain regions.
        components (list): List of PCA components to plot (e.g., ["PC1", "PC2", "PC3"]).
        y (str): The name of the column for the y-axis (categorical values).
        output_dir (Path): Base directory to save the plots.
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).

    Returns:
        None
    """
    control_data = pca_results[pca_results["Brain region"] == "Control"]

    for brain_region in brain_regions:
        # Filter data for the current brain region
        region_data = pca_data[pca_data["Brain region"] == brain_region]
        # Combine control data with the current brain region
        combined_data = pd.concat([control_data, region_data], axis=0).reset_index(drop=True)

        # Create a directory for the current brain region
        region_dir = output_dir / brain_region
        region_dir.mkdir(parents=True, exist_ok=True)

        # Create a grid layout for the PCA components
        nrows = -(-len(components) // ncols)  # Calculate the number of rows needed
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, component in enumerate(components):
            print(f"Generating jitterboxplot for Brain Region: {brain_region}, Component: {component}")

            # Drop rows where the component or y is NaN
            plot_data = combined_data.dropna(subset=[component, y])

            # Create the jitterboxplot
            jitterboxplot(
                data=plot_data,
                x=component,
                y=y,
                hue=hue,
                palette=palette,
                title=f"{component} by {y} ({brain_region})",
                ax=axes[i],  # Pass the corresponding axis
            )

        # Hide any unused subplots
        for j in range(len(components), len(axes)):
            fig.delaxes(axes[j])

        # Save the layout as a single image
        output_path = region_dir / f"PCA_Layout_by_{y.replace(' ', '_')}.pdf"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory

        # Make a 3D plot of the PCA components
        # Ensure the 'hue' column is present in the data
        if hue not in combined_data.columns:
            raise ValueError(f"Column '{hue}' not found in the dataset.")

        # Map the 'hue' values to their corresponding colors using the palette dictionary
        if isinstance(palette, dict):
            color_values = combined_data[hue].map(palette)
        else:
            raise ValueError("Palette must be a dictionary mapping Brain region values to colors.")

        # Make a 3D plot of the PCA components
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            combined_data[components[0]],
            combined_data[components[1]],
            combined_data[components[2]],
            c=color_values,  # Use the mapped colors
            s=50,
            alpha=0.6,
        )
        ax.set_xlabel(components[0])
        ax.set_ylabel(components[1])
        ax.set_zlabel(components[2])
        ax.set_title(f"3D PCA Plot for {brain_region}")
        plt.tight_layout()

        # Save the 3D plot
        output_path_3d = region_dir / f"PCA_3D_{brain_region}.pdf"
        plt.savefig(output_path_3d, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory


# Generate PCA jitterboxplots by brain region


# Example usage
pca_brain_regions = pca_results["Brain region"].unique()  # Get unique brain regions
pca_components = ["PC1", "PC2", "PC3"]  # Specify PCA components to plot

generate_pca_jitterboxplots_by_brain_region(
    pca_data=pca_results,
    brain_regions=pca_brain_regions,
    components=pca_components,
    y="Nickname",  # Categorical column
    output_dir=OUTPUT_DIR / "PCA_Layouts",
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 6),
    ncols=3,  # 3 columns for PC1, PC2, and PC3
)


def generate_pca_jitterboxplots_by_nickname(
    pca_data, nicknames, components, y, output_dir, hue=None, palette="Set2", figsize=(10, 6), ncols=3
):
    """
    Generates jitterboxplots of PCA components by Nickname and its associated control.

    Parameters:
        pca_data (pd.DataFrame): The PCA dataset containing components and metadata.
        nicknames (list): List of unique Nicknames.
        components (list): List of PCA components to plot (e.g., ["PC1", "PC2", "PC3"]).
        y (str): The name of the column for the y-axis (categorical values).
        output_dir (Path): Base directory to save the plots.
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).
        ncols (int, optional): Number of columns in the grid layout. Default is 3.

    Returns:
        None
    """
    for nickname in nicknames:
        print(f"Processing PCA for Nickname: {nickname}")

        # Get the subset of data for the Nickname and its associated control
        subset_data = Config.get_subset_data(pca_data, value=nickname)

        if subset_data.empty:
            print(f"Skipping {nickname} due to empty subset data.")
            continue

        # Sanitize the nickname to remove or replace special characters
        safe_nickname = re.sub(r"[^\w\-_\. ]", "_", nickname)
        nickname_dir = output_dir / safe_nickname
        nickname_dir.mkdir(parents=True, exist_ok=True)

        # Create a grid layout for the PCA components
        nrows = -(-len(components) // ncols)  # Calculate the number of rows needed
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, component in enumerate(components):
            print(f"Generating jitterboxplot for Nickname: {nickname}, Component: {component}")

            # Drop rows where the component or y is NaN
            plot_data = subset_data.dropna(subset=[component, y])

            # Create the jitterboxplot
            jitterboxplot(
                data=plot_data,
                x=component,
                y=y,
                hue=hue,
                palette=palette,
                title=f"{component} by {y} ({nickname})",
                ax=axes[i],  # Pass the corresponding axis
            )

        # Hide any unused subplots
        for j in range(len(components), len(axes)):
            fig.delaxes(axes[j])

        # Save the layout as a single image
        output_path = nickname_dir / f"PCA_Layout_by_{y.replace(' ', '_')}.pdf"
        # Make the directory if it doesn't exist
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory

        # Make a 3D plot of the PCA components
        if len(components) >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                subset_data[components[0]],
                subset_data[components[1]],
                subset_data[components[2]],
                c=subset_data[hue].map(palette) if isinstance(palette, dict) else None,
                s=50,
                alpha=0.6,
            )
            ax.set_xlabel(components[0])
            ax.set_ylabel(components[1])
            ax.set_zlabel(components[2])
            ax.set_title(f"3D PCA Plot for {nickname}")
            plt.tight_layout()

            # Save the 3D plot
            output_path_3d = nickname_dir / f"PCA_3D_{safe_nickname}.pdf"
            plt.savefig(output_path_3d, dpi=300, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory


# Define parameters
nicknames = pca_results["Nickname"].unique()
pca_components = ["PC1", "PC2", "PC3"]

# Generate PCA jitterboxplots by Nickname
generate_pca_jitterboxplots_by_nickname(
    pca_data=pca_results,
    nicknames=nicknames,
    components=pca_components,
    y="Brain region",  # Categorical column
    output_dir=OUTPUT_DIR,
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 6),
    ncols=3,
)


def test_plot_metric(
    data,
    nickname,
    metric,
    y,
    output_dir,
    palette="Set2",
    n_reps=300,
    show_progress=False,
    figsize=(8, 6),
):
    """
    Test function to create a single plot for one metric and one Nickname.

    Args:
        data (pd.DataFrame): The dataset containing metrics and metadata.
        nickname (str): The Nickname to process.
        metric (str): The metric to plot.
        y (str): The name of the column for the y-axis (categorical values).
        output_dir (Path): Directory to save the plot.
        palette (str or list, optional): Color palette for the plot. Default is "Set2".
        n_reps (int, optional): Number of bootstrap replicates. Default is 300.
        show_progress (bool, optional): Whether to show progress for bootstrap replicates. Default is False.
        figsize (tuple, optional): Size of the plot. Default is (8, 6).

    Returns:
        None
    """
    print(f"Processing test plot for Nickname: {nickname}, Metric: {metric}")

    # Get the subset of data for the Nickname and its associated control
    subset_data = Config.get_subset_data(data, value=nickname)

    if subset_data.empty:
        print(f"Skipping {nickname} due to empty subset data.")
        return

    # Sanitize the nickname to remove or replace special characters
    safe_nickname = re.sub(r"[^\w\-_\. ]", "_", nickname)

    # Get brain region for the nickname
    brain_region_values = subset_data["Brain region"].unique()

    if len(brain_region_values) == 1 and brain_region_values[0] == "Control":
        brain_region = "Control"
    else:
        brain_region = next(value for value in brain_region_values if value != "Control")

    # Create the output directory for the Nickname
    nickname_dir = output_dir / brain_region / safe_nickname
    nickname_dir.mkdir(parents=True, exist_ok=True)

    # Separate data for the Nickname and its control
    nickname_data = subset_data[subset_data["Nickname"] == nickname][metric].dropna()
    control_data = subset_data[subset_data["Nickname"] != nickname][metric].dropna()

    if nickname_data.empty or control_data.empty:
        print(f"Skipping metric '{metric}' for Nickname: {nickname} due to insufficient data.")
        return

    # Compute bootstrapped confidence intervals
    nickname_ci = Processing.draw_bs_ci(
        nickname_data.values, func=np.mean, rg=rg, n_reps=n_reps, show_progress=show_progress
    )
    control_ci = Processing.draw_bs_ci(
        control_data.values, func=np.mean, rg=rg, n_reps=n_reps, show_progress=show_progress
    )

    # Compute effect size
    effect_size, effect_size_interval = Processing.compute_effect_size(nickname_ci, control_ci)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    stripplot = sns.stripplot(
        data=subset_data,
        x=metric,
        y=y,
        hue="Brain region",
        palette=palette,
        dodge=False,
        alpha=0.6,
        jitter=True,
        size=5,
        ax=ax,
    )

    # Extract the y positions for the two groups
    y_positions = stripplot.get_yticks()

    # Add bootstrapped confidence intervals as error bars
    ax.errorbar(
        x=[nickname_data.mean(), control_data.mean()],
        y=y_positions[:2],  # Use the first two y positions dynamically
        xerr=[
            [nickname_data.mean() - nickname_ci[0], control_data.mean() - control_ci[0]],
            [nickname_ci[1] - nickname_data.mean(), control_ci[1] - control_data.mean()],
        ],
        fmt="o",
        color="black",
        capsize=5,
        zorder=1,  # Ensure error bars are drawn below the stripplot
    )

    # Add effect size as text on the plot
    ax.text(
        0.8,
        0.5,
        f"Effect Size: {effect_size:.2f}\n95% CI: [{effect_size_interval[0]:.2f}, {effect_size_interval[1]:.2f}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
    )

    # Add a red "*" if the effect size confidence interval does not include zero
    if (
        effect_size_interval[0] > 0
        and effect_size_interval[1] > 0
        or effect_size_interval[0] < 0
        and effect_size_interval[1] < 0
    ):
        ax.text(
            0.5,
            0.5,
            "*",
            transform=ax.transAxes,
            fontsize=20,
            color="red",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Finalize plot
    # Increase y margins
    ax.margins(y=2)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 2, top=ax.get_ylim()[1] + 2)

    ax.set_title(f"{metric} - {nickname} vs Control - {brain_region}", fontsize=14)
    ax.set_xlabel(metric)
    # Remove y-axis label
    ax.set_ylabel("")

    # Hide legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()

    plt.tight_layout()
    plt.show()


# Example usage of the test_plot function
# Test the function with one nickname and one metric
test_plot_metric(
    data=dataset,
    nickname="86639 (LH1990)",  # Replace with an actual nickname from your dataset
    metric="nb_events",  # Replace with an actual metric from your dataset
    y="Nickname",
    output_dir=OUTPUT_DIR,
    palette=Config.color_dict,
    n_reps=300,
    show_progress=True,
    figsize=(8, 3),
)


def plot_metric_group_layouts(
    data,
    nicknames,
    metric_groups,
    y,
    output_dir,
    palette="Set2",
    n_reps=300,
    show_progress=False,
    ncols=3,
    figsize=(8, 6),
):
    """
    Create layouts grouped by metric groups for each Nickname, combining stripplot, confidence intervals, and effect size.

    Args:
        data (pd.DataFrame): The dataset containing metrics and metadata.
        nicknames (list): List of unique Nicknames.
        metric_groups (dict): Dictionary of metric groups (e.g., {"Group Name": [metric1, metric2]}).
        y (str): The name of the column for the y-axis (categorical values).
        output_dir (Path): Base directory to save the plots.
        palette (str or list, optional): Color palette for the plots. Default is "Set2".
        n_reps (int, optional): Number of bootstrap replicates. Default is 300.
        show_progress (bool, optional): Whether to show progress for bootstrap replicates. Default is False.
        ncols (int, optional): Number of columns in the grid layout. Default is 3.
        figsize (tuple, optional): Size of each subplot. Default is (8, 6).

    Returns:
        None
    """
    for nickname in nicknames:
        print(f"Processing layouts for Nickname: {nickname}")

        # Get the subset of data for the Nickname and its associated control
        subset_data = Config.get_subset_data(data, value=nickname)

        if subset_data.empty:
            print(f"Skipping {nickname} due to empty subset data.")
            continue

        # Sanitize the nickname to remove or replace special characters
        safe_nickname = re.sub(r"[^\w\-_\. ]", "_", nickname)

        # Get brain region for the nickname
        brain_region_values = subset_data["Brain region"].unique()

        if len(brain_region_values) == 1 and brain_region_values[0] == "Control":
            brain_region = "Control"
        else:
            brain_region = next(value for value in brain_region_values if value != "Control")

        # Create the output directory for the Nickname
        nickname_dir = output_dir / brain_region / safe_nickname
        nickname_dir.mkdir(parents=True, exist_ok=True)

        for group_name, metrics in metric_groups.items():
            print(f"Processing metric group '{group_name}' for Nickname: {nickname}")

            # Define the output path for the layout
            output_path = nickname_dir / f"{group_name.replace(' ', '_')}_layout.pdf"

            # Check if the plot already exists
            if output_path.exists():
                print(f"Plot for metric group '{group_name}' for Nickname: {nickname} already exists. Skipping...")
                continue

            # Calculate the grid layout
            nrows = -(-len(metrics) // ncols)  # Calculate the number of rows needed
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
            axes = axes.flatten()  # Flatten the axes array for easy iteration

            for i, metric in enumerate(metrics):
                print(f"Processing metric '{metric}' for Nickname: {nickname}")

                # Separate data for the Nickname and its control
                nickname_data = subset_data[subset_data["Nickname"] == nickname][metric].dropna()
                control_data = subset_data[subset_data["Nickname"] != nickname][metric].dropna()

                if nickname_data.empty or control_data.empty:
                    print(f"Skipping metric '{metric}' for Nickname: {nickname} due to insufficient data.")
                    continue

                # Compute bootstrapped confidence intervals
                nickname_ci = Processing.draw_bs_ci(
                    nickname_data.values, func=np.mean, rg=rg, n_reps=n_reps, show_progress=show_progress
                )
                control_ci = Processing.draw_bs_ci(
                    control_data.values, func=np.mean, rg=rg, n_reps=n_reps, show_progress=show_progress
                )

                # Compute effect size
                effect_size, effect_size_interval = Processing.compute_effect_size(nickname_ci, control_ci)

                # Plot raw values as a stripplot
                ax = axes[i]
                stripplot = sns.stripplot(
                    data=subset_data,
                    x=metric,
                    y=y,
                    hue="Brain region",
                    palette=palette,
                    dodge=False,
                    alpha=0.6,
                    jitter=True,
                    size=5,
                    ax=ax,
                )

                # Extract the y positions for the two groups
                y_positions = stripplot.get_yticks()

                # Add bootstrapped confidence intervals as error bars
                ax.errorbar(
                    x=[nickname_data.mean(), control_data.mean()],
                    y=y_positions[:2],  # Adjust y positions for the two groups
                    xerr=[
                        [nickname_data.mean() - nickname_ci[0], control_data.mean() - control_ci[0]],
                        [nickname_ci[1] - nickname_data.mean(), control_ci[1] - control_data.mean()],
                    ],
                    fmt="o",
                    color="black",
                    capsize=5,
                    zorder=1,  # Ensure error bars are drawn below the stripplot
                )

                # Add effect size as text on the plot
                ax.text(
                    0.8,
                    0.5,
                    f"Effect Size: {effect_size:.2f}\n95% CI: [{effect_size_interval[0]:.2f}, {effect_size_interval[1]:.2f}]",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="center",
                    horizontalalignment="center",
                )

                # Add a red "*" if the effect size confidence interval does not include zero
                if (
                    effect_size_interval[0] > 0
                    and effect_size_interval[1] > 0
                    or effect_size_interval[0] < 0
                    and effect_size_interval[1] < 0
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "*",
                        transform=ax.transAxes,
                        fontsize=20,
                        color="red",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

                # Finalize plot
                # Increase y margins
                ax.margins(y=2)
                ax.set_ylim(bottom=ax.get_ylim()[0] - 2, top=ax.get_ylim()[1] + 2)

                ax.set_title(f"{metric} - {nickname} vs Control - {brain_region}", fontsize=14)
                ax.set_xlabel(metric)
                # Remove y-axis label
                ax.set_ylabel("")

                # Hide legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

            # Hide any unused subplots
            for j in range(len(metrics), len(axes)):
                fig.delaxes(axes[j])

            # Save the layout for the metric group
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved layout for metric group '{group_name}' for Nickname: {nickname} to {output_path}")


# Define parameters
nicknames = dataset["Nickname"].unique()
metrics = metric_names  # List of all metrics


# Generate layouts for all metric groups
plot_metric_group_layouts(
    data=dataset,
    nicknames=nicknames,
    metric_groups=metric_groups,  # Dictionary of metric groups
    y="Nickname",
    output_dir=OUTPUT_DIR,
    palette=Config.color_dict,
    n_reps=300,
    show_progress=False,
    ncols=3,  # Number of columns in the grid layout
    figsize=(8, 6),  # Size of each subplot
)


def plot_all_pca_components_sorted(pca_data, components, y, hue, palette="Set2", figsize=(10, 6), output_path=None):
    """
    Creates a single plot for all PCA components, sorted by median value and colored by Brain region.

    Parameters:
        pca_data (pd.DataFrame): The PCA dataset containing components and metadata.
        components (list): List of PCA components to plot (e.g., ["PC1", "PC2", "PC3"]).
        y (str): The name of the column for the y-axis (categorical values).
        hue (str): The name of the column for color grouping.
        palette (str or dict, optional): Color palette for the plot. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).
        output_path (str or Path, optional): Path to save the plot. Default is None.

    Returns:
        None
    """
    # Ensure the y-axis column is categorical and sorted by median value of the first component
    sorted_categories = pca_data.groupby(y, observed=True)[components[0]].median().sort_values(ascending=False).index
    pca_data[y] = pd.Categorical(pca_data[y], categories=sorted_categories, ordered=True)

    # Create a grid layout for the PCA components
    ncols = 3  # Number of columns in the grid layout
    nrows = -(-len(components) // ncols)  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, component in enumerate(components):
        print(f"Creating jitterboxplot for PCA component: {component}")
        ax = axes[i]

        # Drop rows where the component or y is NaN
        plot_data = pca_data.dropna(subset=[component, y])

        # Create the jitterboxplot
        jitterboxplot(
            data=plot_data,
            x=component,
            y=y,
            hue=hue,
            palette=palette,
            title=f"Jitterboxplot of {component} by {y}",
            ax=ax,
        )

    # Hide any unused subplots
    for j in range(len(components), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved PCA components plot to {output_path}")

    plt.show()


# Example usage
pca_components = ["PC1", "PC2", "PC3"]  # Specify PCA components to plot
output_path = OUTPUT_DIR / "All_PCA_Components_Sorted.pdf"

plot_all_pca_components_sorted(
    pca_data=pca_results,
    components=pca_components,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 6),
    output_path=output_path,
)


def plot_individual_pca_components(pca_data, components, y, hue, palette="Set2", figsize=(10, 6), output_dir=None):
    """
    Creates individual plots for each PCA component, sorted by median value and colored by Brain region.

    Parameters:
        pca_data (pd.DataFrame): The PCA dataset containing components and metadata.
        components (list): List of PCA components to plot (e.g., ["PC1", "PC2", "PC3"]).
        y (str): The name of the column for the y-axis (categorical values).
        hue (str): The name of the column for color grouping.
        palette (str or dict, optional): Color palette for the plot. Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (10, 6).
        output_dir (Path, optional): Directory to save the plots. Default is None.

    Returns:
        None
    """
    # Ensure the output directory exists
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for component in components:
        print(f"Creating jitterboxplot for PCA component: {component}")

        # Drop rows where the component or y is NaN
        plot_data = pca_data.dropna(subset=[component, y])

        # Sort the y-axis categories by the median value of the current component
        sorted_categories = plot_data.groupby(y, observed=True)[component].median().sort_values(ascending=False).index
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_categories, ordered=True)

        # Create the plot with the correct axis
        fig, ax = plt.subplots(figsize=figsize)
        jitterboxplot(
            data=plot_data,
            x=component,
            y=y,
            hue=hue,
            palette=palette,
            title=f"Jitterboxplot of {component} by {y}",
            ax=ax,  # Pass the created axis
        )

        # Save the plot if an output directory is provided
        if output_dir:
            output_path = output_dir / f"{component}_Jitterboxplot.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot for {component} to {output_path}")

        plt.close(fig)  # Close the figure to free memory


# Example usage
pca_components = ["PC1", "PC2", "PC3"]  # Specify PCA components to plot
output_dir = OUTPUT_DIR / "Individual_PCA_Components"

plot_individual_pca_components(
    pca_data=pca_results,
    components=pca_components,
    y="Nickname",  # Categorical column
    hue="Brain region",  # Optional grouping column
    palette=Config.color_dict,
    figsize=(10, 30),
    output_dir=output_dir,
)
