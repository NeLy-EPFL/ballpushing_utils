from utils_behavior import HoloviewsTemplates
import pandas as pd
import holoviews as hv
import panel as pn
import Config
from pathlib import Path

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

dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# Configuration section
OUTPUT_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics"
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
    "major_event",
    "major_event_time",
    "major_event_first",
    "insight_effect",
    "insight_effect_log",
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
    "insight_effect",
    "insight_effect_log",
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
# jitterboxplot(
#     data=dataset,
#     x="nb_events",  # Numeric column
#     y="Simplified Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     title="Jitterboxplot of Number of Events by Brain Region",
# )

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
            output_path = region_dir / f"{group_name.replace(' ', '_')}.png"
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
        output_path = output_dir / f"{metric.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory


# Example usage of the jitterboxplot function
# explot = jitterboxplot(
#     data=Combined,
#     x="nb_events",  # Numeric column
#     y="Simplified Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     title="Jitterboxplot of Number of Events by Brain Region",
# )

# plt.savefig("jitterboxplot.png", dpi=300, bbox_inches="tight")

# explot = create_layout(
#     data=Combined,
#     metrics=metric_groups[group],
#     y="Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     figsize=(10, 6),
# )

# # Save as png
# plt.savefig("exLayout.png", dpi=300, bbox_inches="tight")

# brain_regions = dataset["Brain region"].unique()  # Get unique brain regions
# generate_layouts_by_brain_region(
#     data=dataset,
#     brain_regions=brain_regions,
#     metric_groups=metric_groups,
#     y="Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     figsize=(10, 6),
#     ncols=3,
# )

# Generate jitterboxplots for all metrics

# generate_jitterboxplots_for_all_metrics(
#     data=dataset,
#     metrics=metric_names,
#     y="Nickname",  # Categorical column
#     hue="Brain region",  # Optional grouping column
#     palette=Config.color_dict,
#     figsize=(10, 30),
#     output_dir=OUTPUT_DIR / "All_Metrics",
# )

# Load the PCA results

pca_results = pd.read_feather("/home/durrieu/ballpushing_utils/src/Plotting/pca_with_metadata.feather")


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
        output_path = region_dir / f"PCA_Layout_by_{y.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory


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
