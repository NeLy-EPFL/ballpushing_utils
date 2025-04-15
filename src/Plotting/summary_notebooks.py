from utils_behavior import HoloviewsTemplates
import pandas as pd
import holoviews as hv
import panel as pn
import Config

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
    "success_direction",
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


# First use case: generate one jitterboxplot for a single metric

# metric = "nb_events"

# # Create a Holoviews object
# jitterboxplot = HoloviewsTemplates.jitter_boxplot(
#     data=dataset,
#     metric=metric,
#     kdims="Simplified Nickname",
#     control=controls,
#     hline="boxplot",
#     metadata=metadata_names,
#     groupby="Brain region",
#     render="grouped",
#     colorby="Brain region",
#     color_palette=Config.color_dict,
# )

# # Save as html

# hv.save(jitterboxplot, "jitterboxplot.html", fmt="html")


# Layout
# Define a function to create plots for a given metric group and Brain region
def create_plots_for_metric_group(metric_group_name, metrics, dataset, brain_regions, metadata_names):
    plots = []
    for brain_region in brain_regions:
        print(f"Processing Brain region: {brain_region}")

        # Filter dataset for the current Brain region + Control
        subset = dataset[dataset["Brain region"].isin([brain_region, "Control"])]

        # Further filter to include only Simplified Nicknames associated with the current Brain region
        associated_nicknames = subset["Simplified Nickname"].unique()
        subset = subset[subset["Simplified Nickname"].isin(associated_nicknames)]

        # Create a layout of plots for each metric in the group
        brain_region_plots = []
        for metric in metrics:
            plot = HoloviewsTemplates.jitter_boxplot(
                data=subset,
                metric=metric,
                kdims="Simplified Nickname",
                control=controls,
                hline="boxplot",
                metadata=metadata_names,
                groupby=None,
                render="single",
                colorby="Brain region",
                color_palette=Config.color_dict,
            )
            # Ensure each plot has independent axes
            brain_region_plots.append(plot.opts(title=f"{metric} - {brain_region}", shared_axes=False))

        # Combine all plots for the Brain region into a single layout
        plots.append(hv.Layout(brain_region_plots).opts(title=f"Brain Region: {brain_region}", shared_axes=False))

    # Combine all Brain region layouts into a single layout
    return hv.Layout(plots).opts(title=f"Metric Group: {metric_group_name}", shared_axes=False)


# Create tabs for each metric group
tabs = []
brain_regions = dataset["Brain region"].unique()  # Get unique Brain regions
for group_name, metrics in metric_groups.items():
    # Generate plots for the metric group
    print(f"Creating plots for metric group: {group_name}")
    layout = create_plots_for_metric_group(group_name, metrics, dataset, brain_regions, metadata_names)
    tabs.append(pn.Column(f"Tab: {group_name}", pn.pane.HoloViews(layout)))

# Create a Panel Tabs layout
dashboard = pn.Tabs(*[(tab[0], tab[1]) for tab in zip(metric_groups.keys(), tabs)])

# Save the dashboard as an HTML file
pn.io.save.save(dashboard, "summary_dashboard.html", embed=True)

print("Dashboard saved as summary_dashboard.html")
